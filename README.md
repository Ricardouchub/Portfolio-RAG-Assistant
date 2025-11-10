
# Asistente RAG para Portfolio Personal

![Estado del Proyecto](https://img.shields.io/badge/Estado-En%20Desarrollo-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Embeddings](https://img.shields.io/badge/Embeddings-BGE--M3-orange)
![Vector Store](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple)
![LangChain](https://img.shields.io/badge/LangChain-Framework-00BFFF?logo=chainlink&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek-LLM-8A2BE2?logo=deepnote&logoColor=white)



Asistente RAG que ingiere la carpeta de proyectos local, construye un índice en **ChromaDB** con embeddings avanzados **BGE‑M3** y responde preguntas a través de un LLM **DeepSeek**.

Para las pruebas iniciales se incluyó únicamente un repositorio personal '[evalcards](https://github.com/Ricardouchub/evalcards)', una biblioteca en Python que genera reportes de evaluación para modelos supervisados. Ingerir este repositorio demuestra cómo el pipeline procesa documentación, código y notebooks de proyectos reales (ver ejemplos de consulta)

## Características

- **Ingesta profunda:** maneja Markdown, scripts Python, PDFs, archivos DOCX, texto plano y jupyter notebooks con granularidad de celda y etiquetado inteligente de metadatos.
- **Segmentación consciente del código:** separa prosa y código; los archivos `.py` y celdas de código se dividen con un algoritmo aware del lenguaje mientras que los documentos de texto se agrupan en chunks más largos.
- **Recuperación robusta:** embeddings normalizados BAAI/bge‑m3 combinados con un recuperador de Margen Máximo de Relevancia (MMR) para obtener contexto diverso y relevante.
- **Respuestas flexibles:** consulta desde la terminal (`main.py`) o mediante una aplicación Dash (`app.py`).

## Arquitectura

1. **Ingesta (`src/ingest.py`):**
   - Recorre la carpeta `portfolio_data/`, ignorando directorios como `.git`, `__pycache__`, etc.
   - Usa loaders dedicados (Docx2txt, NotebookLoader, etc.) para normalizar metadatos y aplica segmentación por tokens diferenciando prosa y código.
   - Crea embeddings con el modelo BGE‑M3 y persiste los vectores en una colección `portfolio` dentro de `chroma_db/`.

2. **Recuperación y generación (`src/main.py`):**
   - Reconstruye la misma tubería de embeddings, abre la colección y recupera los *k* trozos más relevantes usando MMR.
   - Envía el contexto al modelo DeepSeek para generar respuestas fundamentadas.

3. **Interfaz interactiva (`app.py`):**
   - Aplicación web Dash con Bootstrap que replica la experiencia de la CLI.
   - Solicita la API key de DeepSeek en tiempo de ejecución, muestra instrucciones y mantiene un historial de conversación.

## Estructura del proyecto

```text
.
|-- assets/
|   -- custom.css          # Estilos de Dash App
|-- chroma_db/             # Almacén vectorial (generado por ingest.py)
|-- portfolio_data/        # Coloca tus proyectos o repositorios de portafolio
|-- src/
|   |-- ingest.py          # Ingesta: loaders, tokenización, embeddings BGE, persistencia Chroma
|   |-- main.py            # Asistente RAG para terminal (usa DeepSeek vía LangChain)
|-- app.py                 # Aplicación web Dash para exploración interactiva
|-- requirements.txt       # Dependencias de Python
|-- README.md              # Este archivo
```

## Como usarlo

1. **Prepara tus datos:** clona o descomprime el repositorio y coloca tus proyectos de portafolio (a cualquier nivel de profundidad) dentro de `portfolio_data/`. En este ejemplo la carpeta contiene únicamente el proyecto `evalcards`.
2. **Instala las dependencias** (se recomienda Python 3.10 o superior):

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la ingesta** cada vez que modifiques o añadas archivos:

   ```bash
   python src/ingest.py
   ```

   Esto crea o actualiza la base vectorial en `chroma_db/` con la colección `portfolio_data`.

4. **Escoge tu interfaz:**

   - **Asistente en terminal**
     
     ```bash
     python src/main.py
     ```

     Proporciona tu API key de DeepSeek cuando se solicite (o exporta `DEEPSEEK_API_KEY` como variable de entorno). Luego formula preguntas en lenguaje natural.

   - **Aplicación web Dash**
     
     ```bash
     python app.py
     ```

     Abre `http://127.0.0.1:8050` en tu navegador, pega tu API key, elige un modelo y comienza a consultar. La interfaz destaca consejos de configuración y conserva el historial de la conversación.

## Configuración y personalización

- **Ajustes de DeepSeek:**
  - `DEEPSEEK_API_KEY` (opcional) para evitar el prompt inicial.
  - `DEEPSEEK_CHAT_MODEL` y `DEEPSEEK_BASE_URL` permiten usar otros endpoints o modelos.
- **Soporte de loaders:** amplía `LOADER_MAPPING` en `src/ingest.py` para añadir CSV, HTML u otros formatos propietarios; también puedes activar/desactivar la inclusión de salidas en notebooks.
- **Parámetros de segmentación:** ajusta los tamaños y solapes de los chunks para equilibrar el recall frente a la latencia; separa prosa y código según tus necesidades.
- **Comportamiento del recuperador:** modifica `RETRIEVER_K`, `RETRIEVER_FETCH_K` o cambia `search_type` si prefieres una búsqueda por similitud simple en lugar de MMR.
- **Embeddings:** intercambia `EMBEDDINGS_MODEL` por un modelo más ligero en caso de tener recursos limitados, pero se debe ajustar también `main.py` y `app.py`.

## Ejemplo de consultas

- *"Como se instala evalcards?"*
- *"¿Qué métricas soporta evalcards para forecasting?"*
- *"Dame un ejemplo de clasificación multi-label"*

Estos ejemplos muestran cómo se puede preguntar por documentación técnica, flujos de trabajo o detalles específicos de un repositorio indexado.

<img width="800" src="assets/RAG Assistant.png" alt="Main"/>



## Autor

**Ricardo Urdaneta**

[GitHub](https://github.com/Ricardouchub) | [LinkedIn](https://www.linkedin.com/in/ricardourdanetacastro)
