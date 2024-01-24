import torch
import streamlit as st
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from llama_cpp import Llama
from PIL import Image
import google.generativeai as genai


def load_llm(model_type="fylee", **kwargs):
    # if "lmm" in st.session_state:
    #    del st.session_state.lmm
    #    torch.cuda.empty_cache()

    if "llm_fylee" not in st.session_state and model_type == "fylee":
        st.session_state.llm_fylee = Llama(
            model_path=st.session_state.llm_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
        )
    if "llm_google" not in st.session_state:
        st.session_state.llm_google = genai.GenerativeModel("gemini-pro")
    if "llm_openai" not in st.session_state:
        st.session_state.llm_openai = None


def use_llm(prompt, model_type="fylee", **kwargs):
    if model_type == "fylee":
        output = st.session_state.llm_fylee(prompt, max_tokens=512)
        output = output["choices"][0]["text"]
    elif model_type == "google":
        output = st.session_state.llm_google.generate_content(prompt)
        output = output.text
    else:
        pass  # GPT-4
    return output


def load_lmm(model_type="fylee", **kwargs):
    # if "llm" in st.session_state:
    #    del st.session_state.llm
    #    torch.cuda.empty_cache()

    if "lmm" not in st.session_state and model_type == "fylee":
        st.session_state.lmm_fylee = LlavaForConditionalGeneration.from_pretrained(
            st.session_state.lmm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )
        st.session_state.processor_fylee = AutoProcessor.from_pretrained(
            st.session_state.lmm_path
        )
    if "lmm_google" not in st.session_state:
        st.session_state.lmm_google = genai.GenerativeModel("gemini-pro-vision")
    if "lmm_openai" not in st.session_state:
        st.session_state.lmm_openai = None


def use_lmm(prompt, image_file, model_type="fylee", **kwargs):
    image = Image.open(image_file)
    if model_type == "fylee":
        inputs = st.session_state.processor_fylee(
            prompt, image, return_tensors="pt"
        ).to(0, torch.float16)
        output = st.session_state.lmm_fylee.generate(
            **inputs, max_new_tokens=200, do_sample=False
        )
        output = st.session_state.processor.decode(
            output[0], skip_special_tokens=True
        ).split("Assistant:")[1]

    elif model_type == "google":
        output = st.session_state.lmm_google.generate_content([prompt, image])
        output = output.text

    else:
        output = model_type

    return output
