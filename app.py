# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
from PIL import Image

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Text-to-Image Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Load model (cached) – use CPU if no GPU
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()  # Save VRAM
    return pipe

model = load_model()

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("🖼️ Text-to-Image Generator")
    st.markdown("Create images from text using free Stable Diffusion model.")
    st.markdown("**Model:** CompVis/stable-diffusion-v1-4")
    st.markdown("---")
    st.caption("Tip: Be descriptive for better results (e.g., 'A futuristic city at sunset').")
    st.caption("Generation may take 10-60 seconds depending on hardware.")

# ────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────
st.title("🖼️ Generate Images from Text")
st.markdown("Enter a prompt and let AI create visuals powered by Hugging Face's Stable Diffusion.")

prompt = st.text_area(
    "Enter your text prompt:",
    placeholder="A serene mountain landscape with a crystal-clear lake, in the style of digital art.",
    height=100,
    max_chars=500
)

if st.button("✨ Generate Image", type="primary", use_container_width=True):
    if prompt.strip():
        with st.spinner("Generating image... (Stable Diffusion running)"):
            try:
                # Generate
                image = model(prompt).images[0]

                # Display
                st.success("Image generated!")
                st.image(image, caption="Generated Image", use_column_width=True)

                # Download
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button(
                    label="💾 Download PNG",
                    data=buffer,
                    file_name="generated_image.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                st.info("Tip: Try a simpler prompt or check your internet/hardware.")
    else:
        st.warning("Please enter a text prompt first.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Free model from Hugging Face • Supports CPU/GPU")
