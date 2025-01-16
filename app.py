from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema.runnable import RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from operator import itemgetter
import streamlit as st


class ChatCallBackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def load_message():
    # GPT code 내 방식대로 바꿔볼 것
    converted_history = []
    for msg in st.session_state["messages"]:
        if msg["role"] == "human":
            converted_history.append(HumanMessage(content=msg["message"]))
        elif msg["role"] == "ai":
            converted_history.append(AIMessage(content=msg["message"]))
    return converted_history


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


st.title("Document GPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


with st.sidebar:
    key = st.text_input("pls give me ur API Key!")

if key:

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallBackHandler(),
        ],
        api_key=key,
    )

    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["txt", "pdf", "docx"],
        )

    if file:
        retriever = embed_file(file, key)

        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()

        history = load_history()

        message = st.chat_input("Ask anything about your file!")

        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": itemgetter("question")
                    | retriever
                    | RunnableLambda(format_docs),
                    "question": itemgetter("question"),
                    "history": itemgetter("history"),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke({"question": message, "history": history})

    else:
        st.session_state["messages"] = []
