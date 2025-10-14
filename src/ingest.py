import os
from collections import Counter
from typing import List

# --- LangChain loaders ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,     # .docx
    NotebookLoader,     # .ipynb  (markdown + código; outputs opcionales)
)

# --- Splitters (tokens y code-aware) ---
try:
    # Paquete nuevo recomendado
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
except ImportError:
    # Fallback (algunas instalaciones)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from enum import Enum
    class Language(Enum):  # fallback mínimo
        PYTHON = "python"

# --- Vector store + embeddings ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Token counter (tiktoken) ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except Exception:
    # Fallback por caracteres si no tienes tiktoken
    def count_tokens(text: str) -> int:
        return max(1, int(len(text or "") / 4))  # ~4 chars ≈ 1 token (aprox)


# =======================
# Configuración principal
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "portfolio_data")
PERSIST_DIRECTORY = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_db"))
COLLECTION_NAME   = "portfolio"
EMBEDDINGS_MODEL  = "BAAI/bge-m3"  # multilingüe, muy sólido para retrieval
USE_NOTEBOOK_OUTPUTS = False       # pon True si quieres indexar outputs de celdas

# Carpetas a ignorar
EXCLUDE_DIRS = {
    ".git", ".github", "__pycache__", ".ipynb_checkpoints",
    "venv", "env", ".mypy_cache", ".pytest_cache", "node_modules",
    ".ruff_cache", ".cache", "dist", "build", ".next", ".venv"
}

# Tamaños de chunk recomendados
PROSE_CHUNK_SIZE    = 600   # tokens
PROSE_CHUNK_OVERLAP = 100
CODE_CHUNK_SIZE     = 300   # tokens
CODE_CHUNK_OVERLAP  = 80

# Retriever (MMR para diversidad)
RETRIEVER_K          = 6
RETRIEVER_FETCH_K    = 30
RETRIEVER_LAMBDA     = 0.7   # 0=similaridad, 1=diversidad

# Smoke test
RUN_SMOKE_TEST = True
SMOKE_TEST_QUERIES = [
    "¿Dónde explico la arquitectura del proyecto?",
    "función que carga el dataset y hace el split",
    "cómo ejecuto el pipeline de entrenamiento",
]


# =======================
# Loaders por extensión
# =======================
LOADER_MAPPING = {
    ".md": {
        "loader_class": UnstructuredMarkdownLoader,
        "loader_args": {}
    },
    ".py": {
        "loader_class": TextLoader,
        "loader_args": {"encoding": "utf-8", "autodetect_encoding": True}
    },
    ".pdf": {
        "loader_class": PyPDFLoader,
        "loader_args": {}
    },
    ".txt": {
        "loader_class": TextLoader,
        "loader_args": {"encoding": "utf-8", "autodetect_encoding": True}
    },
    ".docx": {
        "loader_class": Docx2txtLoader,
        "loader_args": {}
    },
    ".ipynb": {
        "loader_class": NotebookLoader,
        "loader_args": {
            "include_outputs": USE_NOTEBOOK_OUTPUTS,
            "max_output": None,
            "remove_newline": False,
            "concatenate": False,   # devuelve un doc por celda; mejor para granularidad
        }
    },
}


# =======================
# Funciones principales
# =======================
def load_documents(source_dir: str):
    """Carga documentos soportados desde la carpeta, con auditoría por extensión."""
    from langchain_core.documents import Document  # para type hints ligeros
    documents: List[Document] = []
    by_ext = Counter()
    total_files = 0

    for root, dirs, files in os.walk(source_dir):
        # filtra carpetas ruidosas in-place (afecta al walk)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file_name in files:
            total_files += 1
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext not in LOADER_MAPPING:
                continue

            loader_info = LOADER_MAPPING[file_ext]
            loader_class = loader_info["loader_class"]
            loader_args  = loader_info.get("loader_args", {})
            file_path = os.path.join(root, file_name)

            try:
                loader = loader_class(file_path, **loader_args)
                docs = loader.load()

                # añade metadata útil
                for d in docs:
                    d.metadata.setdefault("source", file_path)
                    d.metadata["ext"] = file_ext
                    # NotebookLoader suele incluir tipo de celda en metadata;
                    # normalizamos a "cell_type" si existe
                    cell_type = d.metadata.get("type") or d.metadata.get("cell_type")
                    if cell_type:
                        d.metadata["cell_type"] = cell_type

                documents.extend(docs)
                by_ext[file_ext] += len(docs)
            except Exception as exc:
                print(f"⚠️  Error al cargar {file_path}: {exc}")

    print(f"🔎 Archivos escaneados: {total_files}")
    if by_ext:
        print("📊 Resumen por extensión (n° de secciones):")
        for ext, count in by_ext.most_common():
            print(f"  {ext:>6}  →  {count}")
    return documents


def build_splitters():
    """Crea splitters por tokens para prosa y código (Python-aware)."""
    prose_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PROSE_CHUNK_SIZE,
        chunk_overlap=PROSE_CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
    )
    try:
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CODE_CHUNK_SIZE,
            chunk_overlap=CODE_CHUNK_OVERLAP,
            length_function=count_tokens,
        )
    except Exception:
        # Fallback si from_language no está disponible
        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CODE_CHUNK_SIZE,
            chunk_overlap=CODE_CHUNK_OVERLAP,
            length_function=count_tokens,
            separators=["\n\n", "\n", " ", ""],
        )
    return prose_splitter, code_splitter


def smart_chunk(documents):
    """Aplica splitter de código a .py y celdas code; prosa al resto."""
    prose_splitter, code_splitter = build_splitters()
    chunks = []
    for d in documents:
        ext = (d.metadata.get("ext") or "").lower()
        cell_type = (d.metadata.get("cell_type") or "").lower()
        if ext == ".py" or cell_type == "code":
            chunks.extend(code_splitter.split_documents([d]))
        else:
            chunks.extend(prose_splitter.split_documents([d]))
    return chunks


def build_embeddings():
    """Inicializa embeddings HuggingFace con normalización (recomendado para BGE)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def persist_chroma(chunks, embeddings):
    """Crea/persiste Chroma y devuelve el retriever MMR listo para usar."""
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    db.persist()

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA,
        },
    )
    return db, retriever


def main():
    print(f"📥 Cargando documentos desde: {SOURCE_DIRECTORY}")
    docs = load_documents(SOURCE_DIRECTORY)
    if not docs:
        print("No se cargaron documentos. Revisa path y dependencias de loaders.")
        return
    print(f"Documentos cargados (secciones): {len(docs)}")

    print("Dividiendo en chunks (prosa vs. código, por tokens)…")
    chunks = smart_chunk(docs)
    print(f"Chunks generados: {len(chunks)}")

    print(f"Creando embeddings: {EMBEDDINGS_MODEL}")
    embeddings = build_embeddings()

    print(f"Persistiendo en Chroma: {PERSIST_DIRECTORY} (colección: {COLLECTION_NAME})")
    db, retriever = persist_chroma(chunks, embeddings)
    print("Ingesta completada.")


if __name__ == "__main__":
    main()