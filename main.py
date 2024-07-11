import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch

@st.cache_resource
def load_resources():
    model_id = './ckpt/Qwen2-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    embeddings = HuggingFaceEmbeddings(model_name='./ckpt/KoSimCSE-roberta-multitask')
    vectorstore = FAISS.load_local("./data/embedding_data", embeddings, allow_dangerous_deserialization=True)
    return tokenizer, model, vectorstore

def generate_response(tokenizer, model, messages, context=""):
    prompt = f"Context: {context}\n\n" if context else ""
    prompt += f"Human: {messages[-1]['content']}\n\nAssistant:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens=2048, do_sample=True, temperature=0.6)
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

def main():
    st.title("Local Chat")
    tokenizer, model, vectorstore = load_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    use_rag = st.sidebar.toggle("Use RAG", value=True)
    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep only the system message
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            print(user_input)

        with st.chat_message("assistant"):
            context = ""
            if use_rag:
                docs = vectorstore.similarity_search(user_input, k=3)
                context = "참고파일(RAG):\n" + "\n".join([doc.page_content for doc in docs])
            response = generate_response(tokenizer, model, st.session_state.messages, context)
            st.write(response)
            print(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
