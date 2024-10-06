from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain_community.chat_models import ChatOllama
from ollama_llm import model_name, get_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # llm = ChatOllama(model=model_name, streaming=True, callbacks=[stream_handler])
        llm = get_chat_model(callbacks=[stream_handler])
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
