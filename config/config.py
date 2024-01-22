import json
import streamlit as st


def load_session_state(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

        st.session_state.llm_path = config["llm"]["path"]
        st.session_state.quant_lib = config["llm"]["quant_lib"]
        st.session_state.chat_prompt = config["llm"]["chat"]["default_prompt"]
        st.session_state.summary_prompt = config["llm"]["summary"]["default_prompt"]

        st.session_state.lmm_path = config["lmm"]["path"]
        st.session_state.captcha_prompt = config["lmm"]["captcha"]["default_prompt"]
