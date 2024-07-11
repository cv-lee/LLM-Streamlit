import streamlit as st
from model import tokenizer, model

def init_page():
    st.header("Local Chat")

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다."
            }
        ]

def ask(tokenizer, model, text):
    st.session_state.messages.append({"role": "user", "content": f"{text}"})
    input_ids = tokenizer.apply_chat_template(
        st.session_state.messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty = 1.1
    )

    answer = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer

def main():
    init_page()
    init_messages()

    if user_content := st.chat_input("Input your question!"):
        with st.spinner("Bot is typing ..."):
            print(user_content)
            answer = ask(tokenizer, model, user_content)
            print(answer)

    for message in st.session_state.messages[1:]:  # Skip the system message
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    


if __name__ == "__main__":
  main()
