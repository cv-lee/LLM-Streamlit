import torch
import streamlit as st
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from llama_cpp import Llama
from PIL import Image


def load_llm(**kwargs):
    if "lmm" in st.session_state:
        del st.session_state.lmm
        torch.cuda.empty_cache()

    if "llm" not in st.session_state:
        st.session_state.llm = Llama(
            model_path=st.session_state.llm_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
        )


def use_llm(prompt, **kwargs):
    output = st.session_state.llm(prompt, max_tokens=512)
    generated_text = output["choices"][0]["text"]
    return generated_text


def load_lmm(**kwargs):
    if "llm" in st.session_state:
        del st.session_state.llm
        torch.cuda.empty_cache()

    if "lmm" not in st.session_state:
        st.session_state.lmm = LlavaForConditionalGeneration.from_pretrained(
            st.session_state.lmm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )
        st.session_state.processor = AutoProcessor.from_pretrained(
            st.session_state.lmm_path
        )


def use_lmm(prompt, image_file):
    image = Image.open(image_file)
    inputs = st.session_state.processor(prompt, image, return_tensors="pt").to(
        0, torch.float16
    )
    output = st.session_state.lmm.generate(
        **inputs, max_new_tokens=200, do_sample=False
    )
    output = st.session_state.processor.decode(
        output[0], skip_special_tokens=True
    ).split("Assistant:")[1]
    return output
