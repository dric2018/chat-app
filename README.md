# Chat with PDF using vLLM + LangChain
A minimalistic, high-performance RAG application designed to translate natural language into Safe SQL to monitor election results, rankings, turnout, and more.

## A - Description
This is a `chat with your document` app implemented as a solution to the set of challenges for the AI Engineer position. Below are the overall objectives:

 - Answer factual questions grounded in the PDF,
 - Compute aggregations/rankings,
 - Generate charts on demand,
 - Progressively improve robustness, safety, and production readiness.

The typical workflow can be described as follows: _the user types a question and the assistant returns an answer derived `only from the
PDF`._

### Technical constraints and key criteria:
The user:
- gets answers based `only on the PDF content`.
- can ask for counts, rankings, and summaries.
- can request a chart (e.g., histogram/bar chart) and receive it inline.

If the information is not in the dataset, the assistant
clearly says so.

- The PDF is the only allowed data source.
- `Section B` provides reproducible setup instructions, while execution instructions are provided under each level.
- We do our best to cap query results and implement protection against runaway queries.

Data handling:
- PDF extraction must handle repeated headers/footers, page numbers, broken
lines, and tables.
- Ensure entity normalization (e.g., accents/casing) is documented.

Security:
- We do our best not to allow destructive DB operations.
- We do our best not to execute arbitrary code from the model.
- [Addition] 🔒 100% Local/internal - All processing happens on the target machine, no data leaves and no external API calls.

Additional constraints (language):
- The overall (main) language used in this project is English. However, we make sure natural language requests can be sent in French...but processed in English in the backend.

## B - Setup and Execution Instructions

### Project Structure
TBA
### Target machine specs
- Dev: Macbook M4
- (Cloud) GPU resource: RTX 4090 (24 GB) and A100 (40 GB for benchmarking)
- Storage resource: xx GB
- Tested OS: Ubuntu/Debian, Darwin (Mac)

PS: The target environment is either GPU- or CPU-based and is not conditioned on the dev environment. Runnning the app on CPU is only for debuggin and benchmarking purposes as it is a suboptimal way of deploying LLM-based apps.

### Stack (Main)
- **Docker**: Containerization tool to package the entire stack for consistent deployment across environments.
#### LLM Orchestration
- **vLLM**: High-throughput engine for serving LLMs with optimized memory management.
- **LangChain**: Orchestration framework that links your prompts, retrieval logic, and LLM calls.

#### Vector DB
##### Motivation for DB choice between Chroma DB, SQLite and Qdrant

|Feature| 	Qdrant|	ChromaDB| SQLite (vec)|
|---|---|---|---|
|Performance|High-performance, Rust-based vector database built for horizontal scaling, advanced filtering, and high-concurrency production RAG.|Slower (Python/pysqlite overhead). Developer-friendly, Python-native vector store designed for rapid prototyping and local experimentation with minimal setup.|Good for small sets; struggles at scale. Minimalist extension that adds vector search to the familiar SQLite engine|
|Monitoring/Observability|Has a built-in Prometheus metrics endpoint.|Limited native observability.|None (requires custom wrappers).|
|Docker Integration|Excellent; official stable images.|Occasional versioning/dependency headaches.| versioning/dependency headaches.	Easy, but it's just a file.|

PS: I would have gone with `Qdrant` as the database but the SQL requirements as well as the tight deadline make it non feasible. We will, therefore, go with `SQLite (vec) `.

#### Data/Metrics Visualization

- **Prometheus**: Time-series database that scrapes and stores metrics from vLLM and your Python app.
- **Grafana**: Visualization platform to view the JSON dashboard and monitor your system health.
- **MLflow**: Lifecycle platform for tracking experiments, managing model versions, and evaluating RAG prompts.


#### Frontend
- **Streamlit**: Python-based web framework for building a clean chat interface with minimal frontend code.

- **Open WebUI**: Feature-rich, self-hosted AI platform with built-in user management, RAG support, and a ChatGPT-like interface.

We have implemented a `Streamlit` UI to meet the minimal requirements, but went ahead with `Open Web UI` as a desired ChatGPT-like UI.

#### General Backend
- **Python >=3.12**: The primary programming language used for logic, data processing, and glue code.
- **pdfblumber**: Surgical tool for Digital PDFs which doesn't "guess" where a character is but asks the PDF file for the exact X/Y coordinates of every letter and line. 

    - We refrain from using vision-language models (VLMs) like Qwen-VL, Dots OCR, DeepSeek OCR or any LM-based OCR model, to avoid unwanted "hallucinations" of the text based on pixels. 
    - In our case where each decimal point/digit matters, `pdfplumber` (or any of its likes) seems more accurate because it extracts the raw data.
    - We also do not want to allocate additional compute resources to run VLMs in addition to our base LLM(s). However, we do acknowledge VLMs as a more generic way of handling various PDF layouts out of the box
    - See [notebooks/pdf_data_extraction.ipynb](notebooks/pdf_data_extraction.ipynb) for more details about our experiments with the target dataset and how the data extraction pipeline was tailored to the subject matter given we had only one PDF to consider as data source.

### Setup
If Python is not already installed oin the machine, you can do so by running the following commands:
```bash
$ sudo apt-get update && sudo apt-get install -y python3
# it can be installed on Mac OS X using the `brew install python` command 
# or check the guide from https://docs.python-guide.org/starting/install3/osx/
```

Crete virtual environment (venv) on host machine
```bash
$ python3 -m venv .venv # creating the venv and installing project dependencies
$ source .venv/bin/activate && pip install . # then run this to install packages within venv
```

#### Build stack: 

The LLM orchestration dependencies can be installed by running the `init.py` script as follows:
```bash
(.venv) $ python3 src/init.py # run the init script
# this will create the docker containers for running the chat-app
# it will also create and populate the database by ingesting the document data.
```

Once the Stack is up and running, you will be able to access each service via:

|Service| 	External URL (via Nginx)|	Internal URL (within Docker)|
|------|----|---|

|Streamlit UI | http://localhost:8080/ | http://streamlit-app:8501|
|------|----|---|

|vLLM API|	http://localhost:8080/v1|	http://vllm:8000/v1|
|------|----|---|

|vLLM Metrics|	http://localhost:8080/metrics|	http://vllm:8000/metrics|
|------|----|---|

|Grafana	|http://localhost:8080/grafana/	|http://grafana:3000|
|------|----|---|

|Prometheus	|http://localhost:9090 (direct)	|http://prometheus:9090|
|------|----|---|

|ChromaDB	|N/A (Internal only)|	http://chroma-db:8000
|------|----|---|

## 

### Execution
#### Typical workflow
When you open the app, you’re greeted by a clean Streamlit interface where you upload your PDF via a sidebar. Behind the scenes, the app instantly "reads" the document, slices it into manageable chunks, and stores them in ChromaDB so it can remember the content without overloading the AI's memory. You then type a question into the chat box—like "Summarize the risks mentioned on page 5"—and the system performs a high-speed search to pull out the exact sentences needed to answer you. This relevant context is handed off to the vLLM engine, which "thinks" through the data and streams a response back to your chat window word-by-word. While this happens, the Nginx proxy ensures the connection stays stable, and Grafana tracks your MacBook M4's CPU usage to show you exactly how hard the hardware is working to generate those answers.

TBA

### Level 1: Text-to-SQL Agent 

#### Data base considerations
For a stable, high-performance RAG workflow involving electoral data, the proposed schema should be split into structural tables (for precise SQL filtering) and vector tables (for semantic search).
Since our data involves multi-line cells and merged cells, using a normalized relational structure is the most reliable way to prevent the "semantic drift" that happens in raw text RAG.

1. Relational Schema (SQLite)
This structure allows you to answer precise questions like "What was the turnout in commune X?" or "How many votes did candidate Y get?"

```sql
-- TURNOUT & LOCALITY (One row per commune/precinct)
CREATE TABLE turnout (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region TEXT,
    circonscription TEXT,
    commune TEXT,
    registered INTEGER,
    votants INTEGER,
    expressed INTEGER,
    invalid_ballots INTEGER,
    participation_rate REAL, -- e.g., 65.4
    abstention_rate REAL
);

-- CANDIDATES & RESULTS (One row per candidate, per commune)
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turnout_id INTEGER,
    candidate_name TEXT,
    party_group TEXT,
    votes_count INTEGER,
    votes_percent REAL,
    is_winner BOOLEAN DEFAULT 0, -- Set 1 for the highest votes_count in that commune
    FOREIGN KEY(turnout_id) REFERENCES turnout(id)
);

-- Creating indexed for query optimization

-- Speed up Ranking queries (Top candidates)
CREATE INDEX idx_results_votes ON results(votes DESC);

-- Speed up Aggregation (Votes per Party)
CREATE INDEX idx_results_party ON results(party);

-- Speed up Locality lookups
CREATE INDEX idx_turnout_commune ON turnout(commune);
```

2. Vector Schema (sqlite-vec / vss)
For RAG, we need a "contextual chunk" table. Instead of embedding raw rows, we will embed a human-readable summary of the table row.

```sql
-- VECTOR SEARCH TABLE
CREATE TABLE results_embeddings (
    row_id INTEGER PRIMARY KEY,
    content_summary TEXT, -- e.g., "In Commune X, Candidate Y (Party Z) received 450 votes (12%)."
    embedding F32_VEC(1536) -- size can be adjusted (e.g., 1536 for OpenAI, 768 for HuggingFace)
);
```


#### 🛡️ Adversarial Safety & Guardrail Tests (Requirement C)
To ensure the system is production-ready and resistant to malicious prompts, run the following test cases in the chat interface.

#### Acceptance Questions
These questions verify the Aggregation, Ranking, and Charting logic:
- Aggregation: "How many seats did [Party Name] win?"
- Ranking: "Top 10 candidates by score in [Region X]."
- Metrics: "What was the average participation rate by region?"
- Charts: "Show me a histogram of winners by party." (Triggers the Plotly auto-viz component).

#### Policy for Unanswerable Questions 
If the query cannot be resolved via the election_results table:
- Response: "Not found in the provided PDF dataset."
- Explanation: The agent will state which filters (e.g., specific Candidate or Region) were attempted.
- Suggestion: "Try rephrasing with a valid Region name from the list."

### Level 2: Hybrid Router (SQL + RAG for fuzziness, narrative, grounding)
TBA

## Level 3: Improved Agentic (clarification + disambiguation + multi-step)
TBA

### Level 4: Advanced (observability + evaluation + reliability)
TBA

### ⚠️ Troubleshooting
TBA 
# Credits and Acknowledgement
- The project was implemented with partial assistance from LLMs (Qwen3-8b, Gemma-3-12b)