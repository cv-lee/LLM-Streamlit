import json
import deepl
import streamlit as st
import threading
import google.generativeai as genai


def load_session_state(config_path, secret_path=None):
    with open(config_path, "r") as f:
        config = json.load(f)

        st.session_state.llm_path = config["llm"]["path"]
        st.session_state.quant_lib = config["llm"]["quant_lib"]
        st.session_state.chat_prompt = config["llm"]["chat"]["default_prompt"]
        st.session_state.summary_prompt = config["llm"]["summary"]["default_prompt"]

        st.session_state.lmm_path = config["lmm"]["path"]
        st.session_state.captcha_prompt = config["lmm"]["captcha"]["default_prompt"]

        st.session_state.model_lock = threading.Lock()

    if secret_path:
        with open(secret_path, "r") as f:
            secret = json.load(f)
            st.session_state.deepl = deepl.Translator(secret["deepl"])
            st.session_state.google_ai = genai.configure(
                api_key=secret["google_generative_language_client"]
            )
