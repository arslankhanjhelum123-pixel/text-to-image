# app.py
import streamlit as st
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
from PIL import Image

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="CPU-Friendly Text-to-Image",
    page_icon="🖼️✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Load lightweight model (CPU-friendly)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Use a small distilled SD variant that runs decently on CPU
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd",                  # ~246 MB, distilled SD 1.5
        torch_dtype=torch.float32,          # CPU → float32 (float16 often crashes on CPU)
        safety_checker=None,                # Optional: disable if you don't need NSFW filter
        requires_safety_checker=False
    )
    pipe = pipe.to("cpu")                   # Explicitly force CPU
    pipe.enable_attention_slicing()         # Huge memory saver
    pipe.enable_sequential_cpu_offload()    # Even more memory efficient (trades speed for RAM)
    return pipe

model = load_model()

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("🖼️ CPU Text-to-Image Demo")
    st.markdown("Generates images from text using a **lightweight** model (no GPU needed).")
    st.markdown("**Model:** segmind/tiny-sd (distilled Stable Diffusion)")
    st.markdown("---")
    st.caption("Expect 20–120 seconds per image on typical laptops.")
    st.caption("Lower steps = faster, but lower quality.")
    st.caption("Tip: Use detailed English prompts for best results.")

# ────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────
st.title("🖼️ Text-to-Image Generator (CPU Only)")
st.markdown("Type a description and create an image — works on regular laptops without graphics card!")

col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area(
        "Your prompt:",
        placeholder="A cozy cabin in snowy mountains at sunrise, digital painting style, detailed",
        height=120,
        max_chars=400
    )

with col2:
    num_steps = st.slider("Inference steps (quality vs speed)", 15, 50, 28, step=1)
    guidance = st.slider("Guidance scale (how closely it follows prompt)", 3.0, 12.0, 7.5, step=0.5)

if st.button("✨ Generate Image", type="primary", use_container_width=True):
    if prompt.strip():
        with st.spinner(f"Generating on CPU... (this may take 30–90 seconds)"):
            try:
                # Generate with tuned parameters for CPU speed
                image = model(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    height=512,
                    width=512
                ).images[0]

                st.success("Image ready!")
                st.image(image, caption="Generated Image", use_column_width=True)

                # Download
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button(
                    label="💾 Download PNG",
                    data=buffer,
                    file_name="cpu_generated_image.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Generation error: {str(e)}")
                st.info("Try fewer steps (15–20) or a shorter prompt if it fails.")
    else:
        st.warning("Please write a prompt first.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Model: segmind/tiny-sd • Runs fully on CPU • Free & open-source")
