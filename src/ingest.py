import os
from collections import Counter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,          # <- para .docx (alternativa: UnstructuredWordDocumentLoader)
    NotebookLoader,          # <- para .ipynb (incluye markdown y código)
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
SOURCE_DIRECTORY = "./portfolio_data"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "portfolio"

# Carpetas a ignorar en el walk
EXCLUDE_DIRS = {
    ".git", ".github", "__pycache__", ".ipynb_checkpoints", "venv", "env",
    ".mypy_cache", ".pytest_cache", "node_modules", ".ruff_cache", ".cache"
}

# --- Custom Loader Configuration ---
# Nota: UnstructuredMarkdownLoader requiere 'unstructured' instalado.
LOADER_MAPPING = {
    ".md":   {"loader_class": UnstructuredMarkdownLoader, "loader_args": {}},
    ".py":   {"loader_class": TextLoader, "loader_args": {"encoding": "utf-8", "autodetect_encoding": True}},
    ".pdf":  {"loader_class": PyPDFLoader, "loader_args": {}},
    ".txt":  {"loader_class": TextLoader, "loader_args": {"encoding": "utf-8", "autodetect_encoding": True}},
    ".docx": {"loader_class": Docx2txtLoader, "loader_args": {}},
    ".ipynb":{"loader_class": NotebookLoader,
              "loader_args": {"include_outputs": True, "max_output": None, "remove_newline": False, "concatenate": False}},
}

def load_documents(source_dir: str):
    """Load all supported documents from the source directory."""
    documents = []
    by_ext = Counter()

    for root, dirs, files in os.walk(source_dir):
        # filtra carpetas ruidosas
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file_name in files:
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in LOADER_MAPPING:
                loader_info = LOADER_MAPPING[file_ext]
                loader_class = loader_info["loader_class"]
                loader_args = loader_info.get("loader_args", {})
                file_path = os.path.join(root, file_name)
                try:
                    # NotebookLoader requiere path de carpeta base para resolver imágenes; pero con path basta
                    loader = loader_class(file_path, **loader_args)
                    docs = loader.load()
                    # añade extensión y fuente a metadata para auditoría
                    for d in docs:
                        d.metadata.setdefault("source", file_path)
                        d.metadata["ext"] = file_ext
                    documents.extend(docs)
                    by_ext[file_ext] += len(docs)
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to load {file_path}: {exc}")
                    continue

    # Resumen por tipo
    if by_ext:
        print("Resumen de carga por extensión:")
        for ext, count in by_ext.most_common():
            print(f"  {ext}: {count} secciones")
    return documents

def main():
    """Run the ingestion pipeline to build the vector store."""
    print(f"Loading documents from {SOURCE_DIRECTORY}...")
    documents = load_documents(SOURCE_DIRECTORY)
    if not documents:
        print("No documents were loaded. Check your source directory and loader configuration.")
        return
    print(f"Loaded {len(documents)} document sections.")

    # Sugerencia: si quieres trocear código de forma distinta, puedes crear otro splitter
    # para .py usando 'from_language(Language.PYTHON, ...)' y aplicar por separado.
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"🧩 Split into {len(texts)} chunks.")

    print(f"Creating embeddings using {EMBEDDINGS_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    print(f"Creating and persisting vector store to {PERSIST_DIRECTORY}...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    db.persist()
    print("Ingestion complete!")

if __name__ == "__main__":
    main()