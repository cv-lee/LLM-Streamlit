import os
import streamlit as st
from config.config import load_session_state
from models import load_lmm, use_lmm
from PIL import Image


# Config
config_path = "./config/config.json"
secret_path = "./config/secrets/secret.json"
load_session_state(config_path, secret_path)

st.set_page_config(
    page_title="AI-Playground",
    page_icon="🛝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Title
st.title("🤖 캡차 해독하기")
st.subheader("", divider="grey")


# Select Model
st.subheader("")
model_type_list = {
    "파이리 (FSI)": "fylee",
    "바드 (Google)": "google",
    "GPT (OpenAI)": "openai",
}
model_type = st.radio(
    label="1️⃣ 모델을 선택하세요",
    options=list(model_type_list.keys()),
    index=0,
    horizontal=True,
)
model_type = model_type_list[model_type]


# Upload Image
st.subheader("")
image_file = st.file_uploader("2️⃣ 이미지를 업로드하세요", type=["png", "jpg"])
if image_file is not None:
    st.image(Image.open(image_file), caption="업로드 이미지", width=200)


# Write prompt
st.subheader("")
prompt = st.text_area("3️⃣ 텍스트를 입력하세요", value="Decode captcha letters in image.")
prompt = st.session_state.captcha_prompt.format(prompt)


# Decoding Captcha
if st.button("캡차 해독"):
    with st.session_state.model_lock:
        load_lmm(model_type)
        with st.spinner("캡차 해독중..."):
            answer = use_lmm(prompt, image_file, model_type=model_type)
            st.success("캡차 해독완료!")
            st.text("해독 결과:")
            st.write(answer)


# Footer
st.subheader("")
st.subheader("")
_, _, _, _, _, _, logo = st.columns(7)
with logo:
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "..", "assets", "fsec_icon.jpg")
    )
    st.image(image, width=150)
