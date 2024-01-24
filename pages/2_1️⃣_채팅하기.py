import os
import streamlit as st

from config.config import load_session_state
from models import load_llm, use_llm
from PIL import Image


# Config
st.set_page_config(
    page_title="AI-Playground",
    page_icon="🛝",
    layout="wide",
    initial_sidebar_state="expanded",
)
config_path = "./config/config.json"
secret_path = "./config/secrets/secret.json"
load_session_state(config_path, secret_path)


# Title
st.title("👻 채팅하기")
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


# Select to use Korean
st.subheader("")
use_korean_list = {"네": True, "아니요": False}
use_korean = st.radio(
    label="2️⃣ 한국어 지원을 희망하시나요?",
    options=list(use_korean_list.keys()),
    index=1,
    horizontal=True,
)
use_korean = use_korean_list[use_korean]

# Write prompt
st.subheader("")
prompt = st.text_area("3️⃣ 텍스트를 입력하세요")
prompt = st.session_state.chat_prompt.format(prompt)


# Text Generation
if st.button("답변 생성"):
    with st.session_state.model_lock:
        load_llm(model_type)
        with st.spinner("답변 생성중..."):
            if use_korean and model_type == "fylee":
                prompt = st.session_state.deepl.translate_text(
                    prompt, target_lang="EN-US"
                ).text
            answer = use_llm(prompt, model_type=model_type)
            if use_korean and model_type == "fylee":
                answer = st.session_state.deepl.translate_text(
                    answer, target_lang="KO"
                ).text
            st.success("답변 생성완료!")
            st.markdown(answer)


# Footer
st.subheader("")
st.subheader("")
_, _, _, _, _, _, logo = st.columns(7)
with logo:
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "..", "assets", "fsec_icon.jpg")
    )
    st.image(image, width=150)
