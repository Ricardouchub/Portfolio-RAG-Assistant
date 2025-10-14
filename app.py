import os

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# --- RAG Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.abspath(os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = "portfolio"
EMBEDDINGS_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHAT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
AVAILABLE_MODELS = [
    {"label": "DeepSeek Chat", "value": "deepseek-chat"},
    {"label": "DeepSeek Reasoner", "value": "deepseek-reasoner"},
]


def _build_embeddings():
    """Mirror ingest/main embedding settings so retrieved vectors are compatible."""
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


rag_ready = os.path.isdir(PERSIST_DIRECTORY)
retriever = None
prompt = None

if rag_ready:
    embeddings = _build_embeddings()
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template="""
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
""",
        input_variables=["context", "question"],
    )


def run_rag_pipeline(question: str, api_key: str, model: str, base_url: str) -> str:
    """Execute the RAG pipeline and return the generated answer."""
    if not rag_ready or retriever is None or prompt is None:
        raise RuntimeError(
            "Vector store not found. Run src/ingest.py before starting the app."
        )

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        streaming=False,
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


def render_history(history):
    """Return dash components representing the query history."""
    if not history:
        return html.P("Aun no hay consultas previas.", className="text-secondary mb-0")

    entries = []
    for item in reversed(history):
        entries.append(
            html.Div(
                [
                    html.Div(item["question"], className="history-question"),
                    html.Div(item["answer"], className="history-answer"),
                ],
                className="history-entry",
            )
        )
    return entries


external_stylesheets = [
    dbc.themes.CYBORG,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Portfolio RAG Assistant"
server = app.server


def build_layout() -> html.Div:
    """Create the Dash layout."""
    hero = dbc.Container(
        [
            html.Div(
                [
                    html.Span(
                        "Portfolio Insights",
                        className="badge bg-primary bg-opacity-50 fs-6 px-3 py-2 mb-3",
                    ),
                    html.H1("Portfolio RAG Assistant", className="hero-title"),
                    html.P(
                        "Consulta tu portafolio de proyectos con busqueda semantica y generacion de respuestas fundamentadas.",
                        className="hero-subtitle",
                    ),
                ],
                className="hero-content",
            ),
        ],
        fluid=True,
        className="hero-section",
    )

    instructions_card = dbc.Card(
        [
            dbc.CardHeader("Indicaciones", className="card-header-glass"),
            dbc.CardBody(
                [
                    html.P(
                        "1. Ejecuta la ingesta (src/ingest.py) para indexar tu portafolio.",
                        className="mb-2",
                    ),
                    html.P(
                        "2. Introduce tu API key de DeepSeek en el campo correspondiente.",
                        className="mb-2",
                    ),
                    html.P(
                        "3. Formula preguntas sobre tus proyectos y conoce respuestas con contexto.",
                        className="mb-0",
                    ),
                ],
                className="card-body-glass",
            ),
        ],
        className="glass-card",
    )

    configuration_card = dbc.Card(
        [
            dbc.CardHeader("Configuracion", className="card-header-glass"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("API Key", className="form-label"),
                                    dbc.Input(
                                        id="api-key-input",
                                        type="password",
                                        placeholder="Ingresa tu API key de DeepSeek",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Modelo", className="form-label"),
                                    dcc.Dropdown(
                                        id="model-select",
                                        options=AVAILABLE_MODELS,
                                        value=DEFAULT_CHAT_MODEL,
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    )
                ],
                className="card-body-glass",
            ),
        ],
        className="glass-card",
    )

    question_card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Label("Pregunta", className="form-label"),
                    dbc.Textarea(
                        id="question-input",
                        placeholder="Ej. Describe la arquitectura del proyecto X",
                        className="question-input",
                    ),
                    dbc.Button(
                        [
                            html.I(className="bi bi-send me-2"),
                            "Preguntar",
                        ],
                        id="ask-button",
                        color="primary",
                        className="mt-3",
                    ),
                ],
                className="card-body-glass",
            ),
        ],
        className="glass-card",
    )

    answer_placeholder = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div("Respuesta", className="section-title"),
                    html.P(
                        "Formula una pregunta para ver la respuesta aqui.",
                        className="text-secondary mb-0",
                    ),
                ],
                className="card-body-glass",
            ),
        ],
        className="glass-card",
    )

    history_card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div("Historial", className="section-title"),
                    html.Div(id="history-list", className="history-list"),
                ],
                className="card-body-glass",
            ),
        ],
        className="glass-card",
    )

    main_content = dbc.Container(
        [
            dcc.Store(id="history-store", data=[]),
            dbc.Alert(
                "No se encontro la base de conocimiento. Ejecuta src/ingest.py antes de usar la aplicacion.",
                color="warning",
                className="mb-4 glass-card text-warning border-0",
                is_open=not rag_ready,
            ),
            instructions_card,
            html.Div(className="my-4"),
            configuration_card,
            html.Div(className="my-4"),
            dbc.Row(
                [
                    dbc.Col(question_card, lg=5, className="mb-4"),
                    dbc.Col(
                        dcc.Loading(answer_placeholder, type="dot", color="#5f9bff", id="answer-card"),
                        lg=4,
                        className="mb-4",
                    ),
                    dbc.Col(history_card, lg=3, className="mb-4"),
                ],
                className="g-4",
            ),
            dbc.Alert(id="feedback", is_open=False, className="mt-3 glass-card border-0"),
        ],
        fluid=True,
        className="pb-5",
    )

    footer = html.Div(
        "Hecho con LangChain, Chroma y DeepSeek.",
        className="footer",
    )

    return html.Div([hero, main_content, footer], id="app-wrapper")


app.layout = build_layout


@app.callback(
    Output("answer-card", "children"),
    Output("feedback", "children"),
    Output("feedback", "color"),
    Output("feedback", "is_open"),
    Output("history-store", "data"),
    Output("history-list", "children"),
    Input("ask-button", "n_clicks"),
    State("question-input", "value"),
    State("api-key-input", "value"),
    State("model-select", "value"),
    State("history-store", "data"),
    prevent_initial_call=True,
)
def handle_question(n_clicks, question, api_key, model, history_data):  # pragma: no cover
    if not rag_ready:
        warning = (
            "No se encontro la base de conocimiento. Ejecuta src/ingest.py antes de usar la aplicacion."
        )
        return (
            dash.no_update,
            warning,
            "warning",
            True,
            dash.no_update,
            dash.no_update,
        )

    if not question or not question.strip():
        return (
            dash.no_update,
            "Escribe una pregunta para continuar.",
            "warning",
            True,
            dash.no_update,
            dash.no_update,
        )

    api_key = (api_key or "").strip()
    if not api_key:
        return (
            dash.no_update,
            "Introduce tu API key de DeepSeek.",
            "danger",
            True,
            dash.no_update,
            dash.no_update,
        )

    try:
        answer_text = run_rag_pipeline(question.strip(), api_key, model, DEFAULT_BASE_URL)
    except Exception as exc:  # pragma: no cover
        return (
            dash.no_update,
            f"Error al generar respuesta: {exc}",
            "danger",
            True,
            dash.no_update,
            dash.no_update,
        )

    answer_card = [
        html.Div("Respuesta", className="section-title"),
        html.Div(dcc.Markdown(answer_text, className="answer-content"), className="mt-2"),
    ]

    history_data = (history_data or []) + [
        {
            "question": question.strip(),
            "answer": answer_text,
        }
    ]
    history_children = render_history(history_data)

    return (
        answer_card,
        "Consulta completada.",
        "success",
        True,
        history_data,
        history_children,
    )


if __name__ == "__main__":
    app.run(debug=True)
