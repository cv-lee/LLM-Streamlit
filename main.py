import warnings
import streamlit as st
from streamlit_option_menu import option_menu
from models import load_llm, use_llm, load_lmm, use_lmm
from config.config import load_session_state

warnings.filterwarnings("ignore")


def main():
    config_path = "./config/config.json"
    load_session_state(config_path)

    st.set_page_config(
        page_title="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        "<h1 style='text-align: center; background-color: #000045; color: #ece5f6'>AI Playground</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; background-color: #000045; color: #ece5f6'>Ask Anything you want</h4>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        model_task = option_menu(
            "Menu",
            ["채팅", "요약", "캡차 해독"],
            icons=["house", "kanban", "bi bi-robot"],
            menu_icon="app-indicator",
            default_index=0,
            styles={
                "container": {"padding": "4!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#fafafa",
                },
                "nav-link-selected": {"background-color": "#08c7b4"},
            },
        )

    st.subheader(f"👻 {model_task}하기", divider="rainbow")
    model_type = st.radio(
        label="모델을 선택하세요",
        options=["Fylee", "GPT-4", "Bard"],
        index=0,
        horizontal=True,
    )
    st.markdown("")

    if model_task == "채팅":
        prompt = st.text_area("텍스트를 입력하세요")
        prompt = st.session_state.chat_prompt.format(prompt)

        if st.button("답변 생성"):
            load_llm()
            with st.spinner("답변 생성중..."):
                answer = use_llm(prompt)
                st.success("답변 생성완료!")
                st.text("답변:")
                st.write(answer)

    elif model_task == "요약":
        prompt = st.text_area("텍스트를 입력하세요")
        prompt = st.session_state.summary_prompt.format(prompt)

        if st.button("답변 생성"):
            load_llm()
            with st.spinner("답변 생성중..."):
                answer = use_llm(prompt)
                st.success("답변 생성완료!")
                st.text("답변:")
                st.write(answer)

    else:
        image_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg"])
        prompt = st.text_area("텍스트를 입력하세요", value="Decode captcha letters in image.")
        prompt = st.session_state.captcha_prompt.format(prompt)

        if st.button("캡차 해독"):
            load_lmm()
            with st.spinner("캡차 해독중..."):
                answer = use_lmm(prompt, image_file)
                st.success("캡차 해독완료!")
                st.text("해독 결과:")
                st.write(answer)


if __name__ == "__main__":
    main()
