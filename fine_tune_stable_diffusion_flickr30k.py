import os
import csv
import torch
import random
from datasets import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import gc
from datetime import timedelta

# Force PyTorch to use more aggressive memory release
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------------- Memory Optimization Functions ----------------------
def free_memory():
    """Force release of unused memory"""
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------- Dataset Preparation ----------------------
def load_flickr30k_dataset(image_folder, caption_file, limit=None):
    image_to_captions = {}
    
    with open(caption_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        
        for row in reader:
            image_name, _, caption = row
            image_path = os.path.join(image_folder, image_name)
            
            if os.path.exists(image_path):
                if image_path not in image_to_captions:
                    image_to_captions[image_path] = []
                image_to_captions[image_path].append(caption)
    
    image_caption_pairs = []
    for image_path, captions in image_to_captions.items():
        random_caption = random.choice(captions)
        image_caption_pairs.append({"image_path": image_path, "caption": random_caption})
        
        if limit is not None and len(image_caption_pairs) >= limit:
            break
    
    print(f"Loaded {len(image_caption_pairs)} image-caption pairs.")
    return Dataset.from_list(image_caption_pairs)

def transform_images(example):
    try:
        image = Image.open(example["image_path"]).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {example['image_path']}, error: {e}")
        raise

    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    example["pixel_values"] = transform(image)
    return example

def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) if isinstance(example["pixel_values"], list) else example["pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    inputs = tokenizer(
        captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }

# ---------------------- Training Function ----------------------
def train():
    # Check if distributed training environment
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    if is_distributed:
        # Initialize process group
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        is_main_process = rank == 0
        print(f"Distributed training initialized successfully: Rank {rank}/{world_size}, Local Rank: {local_rank}")
    else:
        # Single GPU training
        local_rank = 0
        is_main_process = True
        print("Single GPU training mode")
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Free memory before loading components
    free_memory()
    
    model_name = "CompVis/stable-diffusion-v1-4"
    global tokenizer
    
    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    
    if is_main_process:
        print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, 
        subfolder="text_encoder"
    ).to(device)
    
    if is_main_process:
        print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_name, 
        subfolder="vae"
    ).to(device)
    
    if is_main_process:
        print("Loading UNet...")
    # Load UNet and enable gradient checkpointing to save memory
    unet = UNet2DConditionModel.from_pretrained(
        model_name, 
        subfolder="unet"
    ).to(device)
    # Enable gradient checkpointing, significantly reducing memory usage
    unet.enable_gradient_checkpointing()
    
    if is_main_process:
        print("Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Freeze all parameters except U-Net
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Make the entire U-Net trainable
    unet.requires_grad_(True)
    
    unet.train()
    
    # Print trainable parameter count
    if is_main_process:
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in unet.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # Wrap model for distributed training
    if is_distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # Load the Flickr30k dataset with just one caption per image
    dataset = load_flickr30k_dataset("/mnt/data1/dina/flickr30k_images", "/mnt/data1/dina/captions.txt", limit=None)
    dataset = dataset.map(transform_images)
    
    # Batch settings 
    batch_size = 1
    gradient_accumulation_steps = 8  
    
    # Set up distributed sampler (if using distributed training)
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn
    )

    # Add more epochs for the full dataset
    num_epochs = 3
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        num_update_steps_per_epoch += 1
    max_train_steps = num_epochs * num_update_steps_per_epoch
    
    # Optimizer settings - AdamW with weight decay for full training
    if is_distributed:
        optimizer = torch.optim.AdamW(
            unet.module.parameters(),
            lr=1e-6,  # Low learning rate
            weight_decay=1e-2  # Add weight decay to reduce overfitting
        )
    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=1e-6,  # Low learning rate
            weight_decay=1e-2  # Add weight decay to reduce overfitting
        )
    
    # Use linear warmup and cosine decay learning rate schedule
    lr_scheduler = get_scheduler(
        "cosine",  # Cosine decay
        optimizer=optimizer,
        num_warmup_steps=max(100, int(0.05 * max_train_steps)),  # 5% warmup steps
        num_training_steps=max_train_steps
    )

    # Free memory before training
    free_memory()
    
    if is_main_process:
        print(f"Starting training, device: {device}")
        print(f"Updates per epoch: {num_update_steps_per_epoch}")
        print(f"Total training steps: {max_train_steps}")
        print(f"Number of epochs: {num_epochs}")
    
    # Create checkpoint directory
    checkpoint_dir = "./checkpoints"
    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        # Set epoch for sampler in distributed training
        if is_distributed:
            sampler.set_epoch(epoch)
        
        unet.train()
        if is_main_process:
            progress_bar = tqdm(total=num_update_steps_per_epoch)
        
        optimizer.zero_grad()  # Ensure gradients are zero at start
        
        for step, batch in enumerate(train_dataloader):
            if step % 20 == 0:  
                free_memory()
                
            # Move batch data to appropriate device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                # Get text embeddings
                text_embeddings = text_encoder(input_ids)[0]
                
                # Get image latent representations
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            try:
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise) / gradient_accumulation_steps
                
                # Standard backpropagation
                loss.backward()
                
                # Gradient clipping to prevent explosion
                if is_distributed:
                    torch.nn.utils.clip_grad_norm_(unet.module.parameters(), max_norm=5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=5.0)
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # Update weights
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Synchronize all devices (distributed training)
                    if is_distributed:
                        torch.distributed.barrier()
                    
                    # Update progress bar
                    if is_main_process:
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=loss.detach().item() * gradient_accumulation_steps)
                    
                    global_step += 1
                    
                   
                    if is_main_process and global_step % 500 == 0:  
                        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                        os.makedirs(ckpt_path, exist_ok=True)
                        if is_distributed:
                            unet_to_save = unet.module
                        else:
                            unet_to_save = unet
                        unet_to_save.save_pretrained(os.path.join(ckpt_path, "unet"))
                        print(f"Saved checkpoint to {ckpt_path}")
            except Exception as e:
                if is_main_process:
                    print(f"Training step error: {e}")
                free_memory()
                continue
        
        # Save model after each epoch
        if is_main_process:
            print(f"Epoch {epoch} completed")
            epoch_dir = os.path.join(checkpoint_dir, f"epoch-{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            if is_distributed:
                unet_to_save = unet.module
            else:
                unet_to_save = unet
            unet_to_save.save_pretrained(os.path.join(epoch_dir, "unet"))
            print(f"Saved model for Epoch {epoch} to {epoch_dir}")

    # Save final model
    if is_main_process:
        output_dir = "/mnt/data1/dina/fine_tuned_model"
        os.makedirs(output_dir, exist_ok=True)
        if is_distributed:
            unet_to_save = unet.module
        else:
            unet_to_save = unet
        unet_to_save.save_pretrained(os.path.join(output_dir, "unet"))
        print(f"Final model saved to {output_dir}")

    # Clean up distributed environment
    if is_distributed:
        torch.distributed.destroy_process_group()

    # Complete training
    if is_main_process:
        print("Training completed!")            

# ---------------------- Main Entry ----------------------
if __name__ == "__main__":
    # Prevent distributed training on Windows
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.freeze_support()
    
    random.seed(42)
    torch.manual_seed(42)
    
    train()