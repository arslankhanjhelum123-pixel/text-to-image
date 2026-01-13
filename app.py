# app.py
import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO

st.set_page_config(
    page_title="Fast Text to Image (CPU Only)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",           # Official turbo model - 1 to 4 steps only!
        torch_dtype=torch.float32,          # Must be float32 on CPU
        variant="fp16",                     # We'll use fp16 weights but run in fp32
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to("cpu")
    pipe.enable_attention_slicing()
    return pipe

model = load_model()

with st.sidebar:
    st.header("⚡ Lightning Fast Text-to-Image")
    st.success("Runs perfectly on CPU (no GPU needed!)")
    st.markdown("**Model:** SDXL-Turbo (official by Stability AI)")
    st.caption("Generates high-quality images in just 1–4 steps → 8–18 seconds on CPU!")
    st.markdown("---")

st.title("⚡ Text to Image – Blazing Fast on CPU")
st.markdown("Type anything and get a beautiful image instantly — **no GPU required!**")

prompt = st.text_input(
    "Enter your prompt",
    placeholder="A cute baby panda eating bamboo in a misty forest, cinematic, 4k",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3,1])
with col1:
    generate = st.button("🚀 Generate Image", type="primary", use_container_width=True)
with col2:
    steps = st.selectbox("Steps", [1, 2, 4], index=1, help="1 = fastest, 4 = best quality")

if generate and prompt:
    with st.spinner(f"Generating magic in {steps} step{'s' if steps > 1 else ''}..."):
        image = model(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,        # Must be 0.0 for SDXL-Turbo
            height=512,
            width=512
        ).images[0]

        st.image(image, use_column_width=True)
        
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            "💾 Download Image",
            buf,
            "ai_generated_image.png",
            "image/png",
            use_container_width=True
        )

        st.success(f"Done in {steps} step{'s' if steps > 1 else ''}! ⚡")
