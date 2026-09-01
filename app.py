import hashlib
from pathlib import Path
import streamlit as st
from lib.chat_message_histories.streamlit import StreamlitChatMessageHistory
from lib.text import TextLoader
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.storage.file_system import LocalFileStore
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from pydantic import SecretStr

st.set_page_config(
    page_title="Fullsack GPT Challenge Assignment 06",
    page_icon="🤖",
)


history = StreamlitChatMessageHistory()


class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        history.add_ai_message(self.message)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


with st.sidebar:
    OPENAI_API_KEY = st.text_input(
        label="OpenAI API Key",
        type="password",
    )
    file = st.file_uploader(
        "Upload a text file(.txt only)",
        type=["txt"],
    )
    st.write("https://github.com/animasana/assignment06/blob/main/app.py")


if not OPENAI_API_KEY:
    with st.sidebar:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()


# llm = ChatOpenAI(
#     model="gpt-5.4-mini",
#     streaming=True,
#     callbacks=[
#         ChatCallbackHandler(),
#     ],
#     api_key=SecretStr(OPENAI_API_KEY),
# )

llm = init_chat_model(
    model="openai:gpt-5.6-luna",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    api_key=SecretStr(OPENAI_API_KEY),
)


def sha256_encoder(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def document_id(doc) -> str:
    metadata = "|".join(f"{k}={v}" for k, v in sorted(doc.metadata.items()))
    payload = f"{doc.page_content}\n{metadata}"
    return sha256_encoder(payload)


def add_documents_without_duplicates(vector_store, documents):
    ids = [document_id(doc) for doc in documents]
    existing_ids = {doc.id for doc in vector_store.get_by_ids(ids)}

    new_docs = []
    new_ids = []
    for doc, doc_id in zip(documents, ids):
        if doc_id in existing_ids:
            continue
        new_docs.append(doc)
        new_ids.append(doc_id)

    if not new_docs:
        return []

    return vector_store.add_documents(documents=new_docs, ids=new_ids)


@st.cache_resource(show_spinner=False)
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    Path("./.cache/files/").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.read())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    loader = TextLoader(
        file_path=file_path,
        encoding="utf-8",
    )
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=SecretStr(OPENAI_API_KEY),
    )

    cache_dir = LocalFileStore(root_path=f"./.cache/embeddings/{file.name}")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings_model,
        document_embedding_cache=cache_dir,
        key_encoder='sha256',
    )

    # vectorstore = FAISS.from_documents(
    #     documents=docs,
    #     embedding=cached_embeddings,
    # )

    vector_store = InMemoryVectorStore(embedding=cached_embeddings)

    add_documents_without_duplicates(vector_store, docs)

    retriever = vector_store.as_retriever()

    return retriever


def send_human_message(message):
    st.chat_message("human").markdown(message)
    history.add_user_message(message)


def paint_history():
    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_memory(_):
    return history.messages


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. 
        
            You may also use conversation history to remember user preferences or personal details.
        
            When answering knowledge questions about the document, use ONLY the following context.
        
            If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("Fullstack GPT Challenge Assignment 06")


if file:
    retriever = embed_file(file)

    st.chat_message("ai").write("I'm ready! Ask away!")
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_human_message(message)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": load_memory,
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    history.clear()
