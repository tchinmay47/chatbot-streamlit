import streamlit as st
from streamlit_chat import message as st_message
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# import pyautogui


@st.experimental_singleton
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Hello, this is an AI Chatbot")
# if st.button("Clear Chat"):
#     pyautogui.hotkey("ctrl","F5")

def generate_answer():
    counter=0
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
    # concatenate new user input with chat history (if there is)
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if counter > 0 else input_ids
    # generate a bot response
    #print(bot_input_ids)
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_k=50,
        temperature=0.70,
        pad_token_id=tokenizer.eos_token_id
    )
    #print the output
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": output, "is_user": False})
    counter=counter+1

st.text_input("",placeholder="Type here to chat with the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)  # unpacking
