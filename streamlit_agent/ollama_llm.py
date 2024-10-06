from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import Callbacks

base_url = "http://127.0.0.1:11434"
model_name = "llama3.1:latest"
embedding_model_name = "nomic-embed-text:latest"


def get_chat_model(callbacks: Callbacks = None, model_name: str = model_name, base_url: str = base_url, streaming: bool = False) -> ChatOllama:
    if callbacks is not None:
        return ChatOllama(model=model_name, base_url=base_url, streaming=True, callbacks=callbacks)

    if streaming is True:
        return ChatOllama(model=model_name, base_url=base_url, streaming=True)

    return ChatOllama(model=model_name, base_url=base_url)


def get_embedding_model():
    return OllamaEmbeddings(base_url=base_url, model=model_name)


def get_llm():
    return Ollama(base_url=base_url, model=model_name)