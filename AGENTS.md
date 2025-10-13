# AGENTS.md: Building a Portfolio RAG System

## Project Goal

The primary objective is to create a Retrieval-Augmented Generation (RAG) system using Python. This system will ingest a local folder containing a portfolio of GitHub projects (with diverse file types) and allow a user to ask questions in natural language about the content of these projects.

## Core Tech Stack

* **Orchestration Framework:** LangChain
* **Vector Database:** ChromaDB (local persistence)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
* **LLM:** Placeholder for a local model via Ollama (e.g., `llama3` or `mistral`)

## Agent's Core Directive

Your mission is to construct a complete, two-part Python application:
1.  **`ingest.py`:** A script that processes a local directory (`./portfolio_data`), extracts content from all supported files, generates embeddings, and stores them in a persistent ChromaDB vector store.
2.  **`main.py`:** A script that takes a user's question, uses the vector store to retrieve relevant context, and passes it to an LLM to generate a comprehensive answer.

Follow the execution plan below step-by-step.

---

## Step-by-Step Execution Plan

### **Phase 1: Environment is OPTIONAL and Project Setup**

1.  **Create Project Directory:**
    Initialize the main project folder and subdirectories.

    ```bash
    mkdir rag-langchain
    cd rag-langchain
    mkdir src
    mkdir rag-langchain
    ```
    > **Note for User:** Place all your GitHub project folders inside `portfolio_data`.

2.  **Set Up Python Virtual Environment: (Optional)**
    Create and activate a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate
    # For Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies: (Optional)**
    Create a `requirements.txt` file and then install the necessary libraries.

    ```bash
    # Create requirements.txt
    touch requirements.txt
    ```

    **Populate `requirements.txt` with:**
    ```
    langchain
    langchain-community
    langchain-huggingface
    chromadb
    sentence-transformers
    # Add loaders for specific file types
    pypdf # for .pdf
    python-docx # for .docx
    unstructured # for various types, including .md
    notebook # for .ipynb
    # Recommended for connecting to local LLM
    langchain-ollama
    ```

    **Install from the file:**
    ```bash
    pip install -r requirements.txt
    ```

### **Phase 2: The Ingestion Pipeline (`src/ingest.py`)**

This script will perform the indexing of all your documents.

**Create the file `src/ingest.py` and populate it with the following code:**

```python
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
# 1. Path to the directory containing your portfolio projects
SOURCE_DIRECTORY = "./portfolio_data"
# 2. Path to the directory where the Chroma vector store will be persisted
PERSIST_DIRECTORY = "./chroma_db"
# 3. Name of the embedding model to use
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Custom Loader Configuration ---
# Define how to load different file types
LOADER_MAPPING = {
    ".md": {"loader_class": UnstructuredMarkdownLoader},
    ".py": {"loader_class": TextLoader, "loader_args": {"encoding": "utf-8"}},
    ".pdf": {"loader_class": PyPDFLoader},
    ".txt": {"loader_class": TextLoader, "loader_args": {"encoding": "utf-8"}},
    # Add other file types and their loaders here
    # ".csv": {"loader_class": CSVLoader},
    # ".docx": {"loader_class": Docx2txtLoader},
}

def load_documents(source_dir):
    """
    Loads all documents from the source directory using the specified loaders.
    """
    documents = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1]
            if file_ext in LOADER_MAPPING:
                loader_info = LOADER_MAPPING[file_ext]
                loader_class = loader_info['loader_class']
                loader_args = loader_info.get('loader_args', {})
                file_path = os.path.join(root, file)
                try:
                    loader = loader_class(file_path, **loader_args)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
                    continue
    return documents

def main():
    """
    Main function to run the ingestion pipeline.
    """
    # 1. Load documents
    print(f"Loading documents from {SOURCE_DIRECTORY}...")
    documents = load_documents(SOURCE_DIRECTORY)
    if not documents:
        print("No documents were loaded. Check your source directory and loader configuration.")
        return
    print(f"Loaded {len(documents)} document sections.")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. Create embeddings
    print(f"Creating embeddings using {EMBEDDINGS_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # 4. Create and persist the vector store
    print(f"Creating and persisting vector store to {PERSIST_DIRECTORY}...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
```
---

### **Phase 3: The Query & Generation Pipeline (src/main.py)
This script will use the indexed data to answer questions.

Create the file src/main.py and populate it with the following code:

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3" # Make sure you have pulled this model with Ollama

# --- RAG Chain Setup ---

# 1. Initialize Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# 2. Initialize the Retriever
retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks

# 3. Define the Prompt Template
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

# 4. Initialize the LLM
llm = Ollama(model=LLM_MODEL)

# 5. Create the RAG Chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(query):
    """
    Invokes the RAG chain with a user query and streams the response.
    """
    print(f"\n> Query: {query}\n")
    print("AI Answer: ")
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    print("Portfolio RAG Assistant is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
        ask_question(user_input)
```

---

## How to Run the System

1.  **Place Data:** Ensure all your project files are inside the `portfolio_data` directory.
2.  **Run Ingestion:** Execute the ingestion script **once** to process and index all your files.
    ```bash
    cd src
    python ingest.py
    ```
    This will create a `chroma_db` folder in the root directory.
3.  **Run Main App:** Execute the main script to start asking questions.
    ```bash
    python main.py
    ```

## Success Criteria

The agent's task is complete when:
* Running `ingest.py` successfully creates a `chroma_db` directory populated with vector data without errors.
* Running `main.py` starts an interactive command-line interface.
* Asking a question relevant to the documents in `portfolio_data` (e.g., "What is the purpose of the `main.py` file in Project X?") returns a coherent and factually correct answer based on the file's content.
* Asking a question not covered by the documents results in the model stating it doesn't have enough information.