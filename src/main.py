import os
from getpass import getpass

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_db"))
COLLECTION_NAME = "portfolio"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHAT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _build_embeddings():
    """Mirror ingest.py embedding settings so query vectors match the store."""
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    return HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


if not os.path.isdir(PERSIST_DIRECTORY):
    raise RuntimeError(
        "Vector store not found. Run src/ingest.py before starting the assistant."
    )


deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    deepseek_api_key = getpass("Enter your DeepSeek API key: ")
if not deepseek_api_key:
    raise RuntimeError(
        "A DeepSeek API key is required to run main.py."
    )

chat_model = os.getenv("DEEPSEEK_CHAT_MODEL", DEFAULT_CHAT_MODEL)
base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)

embeddings = _build_embeddings()
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

template = """
### INSTRUCTIONS ###
You are an expert assistant for a software developer's project portfolio.
Answer the user's question based ONLY on the following context.
If the context does not contain the answer, state that you don't have enough information.
Do not make up information. Be concise and precise.

### CONTEXT ###
{context}

### QUESTION ###
{question}

### ANSWER ###
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

llm = ChatOpenAI(
    model=chat_model,
    api_key=deepseek_api_key,
    base_url=base_url,
    temperature=0,
    streaming=True,
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask_question(query: str) -> None:
    """Run the RAG chain for a given user query."""
    print(f"\n> Query: {query}\n")
    print("AI Answer: ")
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("Portfolio RAG Assistant is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "exit":
            break
        ask_question(user_input)
