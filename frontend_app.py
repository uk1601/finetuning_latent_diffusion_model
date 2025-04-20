# frontend_app.py
import streamlit as st
import requests
from PIL import Image
import io
import base64 # To embed CSS
from pathlib import Path # To load CSS file

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/generate/" # Default FastAPI backend URL
DEFAULT_PROMPT = "A photo of an astronaut riding a horse on the moon"
EXAMPLE_PROMPTS = [
    "A photo of an astronaut riding a horse on the moon",
    "A cute cat wearing sunglasses and sitting on a beach chair, photorealistic",
    "Impressionist painting of a sunflower field at sunset",
    "A futuristic cityscape with flying cars, synthwave style",
    "Logo for a coffee shop called 'The Daily Grind', minimalist",
    "Steampunk clockwork owl, intricate details, 4k resolution",
]

# --- Load Custom CSS ---
def load_css(file_name):
    css_path = Path(__file__).parent / file_name
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {css_path}")

# --- Helper Function for Download Button ---
def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨", # Favicon emoji
    layout="wide"
)

# Apply custom CSS
load_css("styles.css")

# --- Page Title ---
st.title("🎨 AI Image Generation Studio")
st.markdown("Bring your text prompts to life! Choose between the original Stable Diffusion model or a fine-tuned version.")

# --- Main Layout (Input Form on Left, Output on Right) ---
col1, col2 = st.columns([1, 2]) # Input column narrower than output column

with col1:
    st.header("⚙️ Generation Settings")

    # Use a form to group inputs and submit button
    with st.form("generation_form"):
        prompt = st.text_area(
            "📝 **Enter your prompt:**",
            height=125,
            # value=DEFAULT_PROMPT,
            help="Describe the image you want to create."
        )

        # Example Prompts Selector
        st.markdown("**💡 Or try an example prompt:**")
        selected_example = st.selectbox(
            "Select an example:",
            options=["-- Select --"] + EXAMPLE_PROMPTS,
            index=0, # Default to '-- Select --'
            label_visibility="collapsed" # Hide label, use markdown above
        )
        # Update prompt if an example is selected (requires rerun, handled by form)
        if selected_example != "-- Select --":
           prompt = selected_example # Overwrite prompt with selected example

        st.markdown("---") # Separator

        selected_models = st.multiselect(
            "🤖 **Choose Model(s):**",
            options=['original', 'fine-tuned'],
            default=['fine-tuned'], # Default selection
            help="Select one or both models to generate images."
        )

        with st.expander("🔧 Advanced Settings"):
            num_steps = st.slider("Inference Steps:", min_value=10, max_value=100, value=30, step=1, help="More steps can improve quality but take longer.")
            guidance = st.slider("Guidance Scale (CFG):", min_value=1.0, max_value=20.0, value=7.5, step=0.1, help="How strongly the image should follow the prompt.")
            seed_value = st.number_input("Seed:", value=42, step=1, help="Use -1 for a random result, or a specific number for reproducibility.")

        # Submit button for the form
        submitted = st.form_submit_button("✨ Generate Image(s)", type="primary")

# --- Output Column ---
with col2:
    st.header("🖼️ Generated Images")
    results = {} # Dictionary to store results {model_type: image_or_error}

    # --- Backend Communication & Display Logic ---
    if submitted:
        if not prompt:
            st.error("❗️ Please enter a prompt.")
        elif not selected_models:
            st.error("❗️ Please select at least one model.")
        else:
            num_selected = len(selected_models)
            st.info(f"⏳ Generating images for {num_selected} model(s): {', '.join(selected_models)}...")

            # Prepare placeholders dynamically based on selection
            placeholders = {}
            display_cols = st.columns(num_selected) if num_selected > 0 else []

            for idx, model_type in enumerate(selected_models):
                if idx < len(display_cols):
                    with display_cols[idx]:
                        placeholders[model_type] = st.empty() # Placeholder for spinner/result

            # Loop through selected models and make requests
            for model_type in selected_models:
                placeholder = placeholders.get(model_type)
                if placeholder:
                    with placeholder: # Display spinner within the specific placeholder
                        with st.spinner(f"Processing with {model_type} model..."):
                            request_data = {
                                "prompt": prompt,
                                "model_type": model_type,
                                "num_inference_steps": num_steps,
                                "guidance_scale": guidance,
                                "seed": seed_value
                            }
                            try:
                                response = requests.post(BACKEND_URL, json=request_data, timeout=300) # 5 min timeout

                                if response.status_code == 200:
                                    try:
                                        image_bytes = response.content
                                        image = Image.open(io.BytesIO(image_bytes))
                                        results[model_type] = image # Store successful image
                                    except Exception as e:
                                        results[model_type] = f"Frontend Error: Failed to decode/display image. {e}"
                                else:
                                    try:
                                        error_data = response.json()
                                        detail = error_data.get("detail", "No specific error message provided.")
                                    except requests.exceptions.JSONDecodeError:
                                        detail = response.text # If not JSON, show raw text
                                    results[model_type] = f"Backend Error (Status {response.status_code}): {detail}"

                            except requests.exceptions.RequestException as e:
                                results[model_type] = f"Connection Error: Failed to connect to backend. {e}"
                            except Exception as e:
                                 results[model_type] = f"Frontend Error: An unexpected error occurred. {e}"

            # --- Display Final Results ---
            st.markdown("---") # Separator before showing results
            st.subheader("Results:")

            if not results:
                 st.warning("No results were generated.")

            final_display_cols = st.columns(num_selected) if num_selected > 0 else []
            for idx, model_type in enumerate(selected_models):
                 if idx < len(final_display_cols):
                    with final_display_cols[idx]:
                        result = results.get(model_type)
                        st.markdown(f"##### {model_type.replace('-', ' ').title()} Model") # Smaller header for result
                        if isinstance(result, Image.Image):
                            st.image(result, caption=f"Seed: {seed_value if seed_value >= 0 else 'Random'}", use_container_width=True)
                            # Add download button
                            st.download_button(
                                label="Download Image",
                                data=image_to_bytes(result),
                                file_name=f"{prompt[:30].replace(' ','_')}_{model_type}_{seed_value}.png",
                                mime="image/png"
                            )
                        elif isinstance(result, str): # Error message
                            st.error(result)
                        else:
                            st.warning("Generation was not completed for this model.")

    else:
         # Initial message when no button clicked yet
         st.info("Configure settings on the left and click 'Generate Image(s)' to start.")