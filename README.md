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
- Tested OS: Unix-based, Darwin (Mac)

PS: The target environment is either GPU- or CPU-based and is not conditioned on the dev environment. 
Runnning the app on CPU was mainly for debugging and benchmarking purposes, but it happend to be sufficient to have a working stack.

### Stack (Main)
- **Docker**: Containerization tool to package the entire stack for consistent deployment across environments.
#### LLM Orchestration
- **vLLM**: High-throughput engine for serving LLMs with optimized memory management.
- **LangChain**: Orchestration framework that links your prompts, retrieval logic, and LLM calls.

#### Data/Metrics Visualization

- **Prometheus**: Time-series database that scrapes and stores metrics from vLLM and your Python app.
- **Grafana**: Visualization platform to view the JSON dashboard and monitor your system health.

#### Frontend
- **Streamlit**: Python-based web framework for building a clean chat interface with minimal frontend code.

#### General Backend
- **Python >=3.12**: The primary programming language used for logic, data processing, and glue code.
- **pdfblumber**: Surgical tool for Digital PDFs which doesn't "guess" where a character is but asks the PDF file for the exact X/Y coordinates of every letter and line. 

    - We refrain from using vision-language models (VLMs) like Qwen-VL, Dots OCR, DeepSeek OCR or any LM-based OCR model, to avoid unwanted "hallucinations" of the text based on pixels. 
    - In our case where each decimal point/digit matters, `pdfplumber` (or any of its likes) seems more accurate because it extracts the raw data.
    - We also do not want to allocate additional compute resources to run VLMs in addition to our base LLM(s). However, we do acknowledge VLMs as a more generic way of handling various PDF layouts out of the box
    - See [notebooks/pdf_data_extraction.ipynb](notebooks/pdf_data_extraction.ipynb) for more details about our experiments with the target dataset and how the data extraction pipeline was tailored to the subject matter given we had only one PDF to consider as data source.

### Setup
Make sure docker is already installed on the host machine. Installation details can be found at [subfuzion/install-docker-ubuntu.md](https://gist.github.com/subfuzion/90e8498a26c206ae393b66804c032b79) on GitHub Gist.

Typically, it can be done by running the following command line:
```sh
$ curl -fsSL https://get.docker.com/ | sh
```

If Python is not already installed oin the machine, you can do so by running the following commands:
```bash
$ sudo apt-get update && sudo apt-get install -y python3.13 python3.13-venv python3.13-dev

# it can be installed on Mac OS X using the `brew install python` command 
# or check the guide from https://docs.python-guide.org/starting/install3/osx/
```

Create a virtual environment (venv) on the host machine
```bash
$ python3.13 -m venv .venv # creating the venv and installing project dependencies
$ source .venv/bin/activate && pip install . # then run this to install packages within venv
```

Once these are successfully installed, you can clone this repository with:

```bash
$ git clone git@github.com:dric2018/chat-app.git # and cd into the chat-app folder
```

#### Build stack: 

The LLM orchestration dependencies can be installed by running the `init.py` script as follows:
```bash
(.venv) $ python3.13 src/init.py # run the init script
# this will create the docker containers for running the chat-app
# it will also create and populate the database by ingesting the document data.
```

Once the Stack is up and running, you will be able to access each service via:

|Service|URL|
|---|---|
|Streamlit UI (via Nginx)| http://localhost:8080/|
|vLLM API   |http://localhost:8000/v1/|
|Grafana	|http://localhost:3000|
|Prometheus	|http://localhost:9090|


### Text-to-SQL Agent 

#### Data base considerations
For a stable, high-performance RAG workflow involving electoral data, the proposed schema should be split into structural tables (for precise SQL filtering) and vector tables (for semantic search).
Since our data involves multi-line cells and merged cells, using a normalized relational structure is the most reliable way to prevent the "semantic drift" that happens in raw text RAG.

Overall workflow:
User question
→ Intent classifier
→ If aggregation → SQL generator
→ Validate SQL
→ Execute
→ Return:
   - short narrative
   - dataframe preview
   - optional chart


#### DuckDB

I. Relational Schema

This structure allows to answer precise questions like "What was the turnout in commune X?" or "How many votes did candidate Y get?"

Data base tables are categorized int:
1. Dimentions: Region, Party, Candidate
2. Facts: Turnout (separate), Results (central)

`dim_party`
```sql
party_id INTEGER PRIMARY KEY
party_name TEXT
party_short TEXT
party_normalized TEXT
```

`dim_candidate`
```sql
candidate_id INTEGER PRIMARY KEY
candidate_name TEXT
candidate_normalized TEXT
party_id INTEGER
```

`fact_results`
```sql
result_id INTEGER PRIMARY KEY
region_id INTEGER
candidate_id INTEGER
votes INTEGER
vote_percent FLOAT
elected BOOLEAN
rank INTEGER
source_page INTEGER
source_row_hash TEXT
```
This fact has indexes:

- region_id
- candidate_id
- party_id (via join)
- elected
- votes DESC

`fact_turnout`
```sql
turnout_id INTEGER PRIMARY KEY
region_id INTEGER
registered INTEGER  -- inscrits
voters INTEGER      -- votants
valid_votes INTEGER -- exprimés
blank_votes INTEGER
null_votes INTEGER
participation_rate FLOAT
source_page INTEGER
```
II. Vector Schema 

For RAG, we need a "contextual chunk" table. Instead of embedding raw rows, we will embed a human-readable summary of the table row.

RAG-ready Table: `fact_results_rag`
```sql
row_id INTEGER PRIMARY KEY
entity_type TEXT  -- result, turnout
entity_id INTEGER
region_id INTEGER
text_chunk TEXT --- not directly extracted from PDF tables but can be used to describe results, e.g., "In Tiapoum, candidate X (RHDP) received Y votes (Z%) and was elected."
embedding FLOAT[1536] --- (dim: 1536/OpenAI, 758 HF)
source_page INTEGER
```

`entity_alias`
```sql
alias TEXT
normalized_alias TEXT
entity_type TEXT
entity_id INTEGER
```

Instead of exposing raw tables, we expose curated views:

```sql
--- winners view
CREATE VIEW vw_winners AS
SELECT
    r.region_name,
    c.candidate_name,
    p.party_name,
    f.votes,
    f.vote_percent
FROM fact_results f
JOIN dim_candidate c USING(candidate_id)
JOIN dim_party p USING(party_id)
JOIN dim_region r USING(region_id)
WHERE f.is_winner = TRUE;

--- seats per party
CREATE VIEW vw_party_seats AS
SELECT
    p.party_name,
    COUNT(*) AS seats
FROM fact_results f
JOIN dim_candidate c USING(candidate_id)
JOIN dim_party p USING(party_id)
WHERE f.is_winner = TRUE
GROUP BY p.party_name;

--- turnout per region
CREATE VIEW vw_turnout AS
SELECT
    r.region_name,
    t.registered,
    t.voters,
    t.valid_votes,
    t.participation_rate
FROM fact_turnout t
JOIN dim_region r USING(region_id);


```

```sql
CREATE TABLE dim_region AS
SELECT * FROM read_parquet('dim_region.parquet');

CREATE TABLE dim_party AS
SELECT * FROM read_parquet('dim_party.parquet');

CREATE TABLE dim_candidate AS
SELECT * FROM read_parquet('dim_candidate.parquet');

CREATE TABLE fact_results AS
SELECT * FROM read_parquet('fact_results.parquet');

CREATE TABLE fact_turnout AS
SELECT * FROM read_parquet('fact_turnout.parquet');

--- We precompute
ALTER TABLE fact_results ADD COLUMN is_winner BOOLEAN;

UPDATE fact_results
SET is_winner = (rank = 1);

```



#### Base LLMs
CPU:
- Qwen/Qwen3-0.6B ( ✅ )
    - MAX_TOKENS = 1024
- Qwen/Qwen3-1.7B ( ✅ )
    - MAX_TOKENS = 1024

- Qwen/Qwen3-4B-Instruct-2507 
- facebook/opt-125m ( ❌ )

#### 🛡️ Adversarial Safety & Guardrail Tests (Requirement C)
To ensure the system is nearly production-ready and resistant to malicious prompts, run the following test cases in the chat interface.

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

### Hybrid Router (SQL + RAG for fuzziness, narrative, grounding)
TBA

### Improved Agentic (clarification + disambiguation + multi-step)
TBA

### Advanced (observability + evaluation + reliability)
TBA

### ⚠️ Troubleshooting
TBA 
# Credits and Acknowledgement
- The project was implemented with partial assistance from LLMs (Qwen3-8b, Gemma-3-12b)