import os
import streamlit as st
import torch
from llama_cpp import Llama
#from ctransformers import AutoModelForCausalLM

#os.environ["CUDA_VISIBLE_DEVICES"]= "0"

model_path = './ckpt/solar-10.7b-instruct-v1.0.Q4_K_M.gguf'
def load_model(model_path):
    model = Llama(
        model_path=model_path,  # Download the model file first
        n_ctx=4096,             # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=4,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=49         # The number of layers to offload to GPU, if you have GPU acceleration available
    )
    '''
    model = AutoModelForCausalLM.from_pretrained(model_path,
        model_type="mistral",
        gpu_layers=50)
    '''
    return model

model = load_model(model_path)
st.title("FSI")
    
prompt = st.text_area("Enter text prompt:")

if st.button("Generate"):
    with st.spinner("Generating..."):
        generated_text = model(prompt)['choices'][0]['text']
        st.success("Text generated successfully!")
        st.text("Generated Text:")
        st.write(generated_text)