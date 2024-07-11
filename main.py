import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model_id = './ckpt/Qwen2-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

def generate_response(tokenizer, model, messages):
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.1)
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

def main():
    st.header("Local Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다."}]

    tokenizer, model = load_model()

    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep only the system message

    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(tokenizer, model, st.session_state.messages)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
