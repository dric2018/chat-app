# Chat with PDF using vLLM + LangChain
A minimalistic, chat application designed to translate natural language into safe queries to learn more about election results, rankings, turnout, and more.

Demo video: https://drive.google.com/file/d/1okolo1f7uRzpQHx6NWg1BpU3mZadJa05/view

## A - Description
This is a `chat-with-your-document` app implemented as a solution to a set of challenges for an AI Engineer position. Below are the overall objectives:

 - Answer factual questions grounded in the PDF,
 - Compute aggregations/rankings,
 - Generate charts on demand,
 - Progressively improve robustness, safety, and production readiness.

The typical workflow can be described as follows: _the user types a question/query and the assistant returns an answer derived `only from the
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
- We implemented various protections against runaway queries.

Data handling:
- Our PDF data extraction strategy handles repeated headers/footers, page numbers, broken
lines, and tables (using the `pdfblumber` library).
- We perform entity normalization for accents/casing/punctuations. This is done using the `normalize_text()` function in the [src/utils.py](src/utils.py)

Security:
- A query validation function prevents the agent from executing destructive DB operations.
- Execution of arbitrary code from the model is yet to be test 🏗️.
- [Addition] 🔒 Local deployment - All processing happens on the target machine, no data leaves and no external API calls.

Additional constraints (language):
- The overall (main) language used in this project is English. However, we make sure natural language requests can be sent in French...but processed in English in the backend.

## B - Setup and Execution Instructions

### Overall Approach
The challenge is structured from Level 1 to Level 4.
#### L1 - Text-to-SQL Agent 
Goal: Build a chat app that answers aggregation / ranking / chart questions by translating the
user request into safe SQL, executing it on a structured dataset extracted from the PDF, and
formatting the result.

#### L2 - Hybrid Router (SQL + RAG for fuzziness, narrative, grounding)
Goal: Improve robustness by adding a hybrid router:
- SQL path for analytics (counts/rankings/charts)
- RAG path for fuzzy lookup, narrative explanations, and grounding.

#### L3 - Improved Agentic (clarification + disambiguation + multi-step)
Goal: Make the assistant behave like a real agent; it should ask clarifying questions when
needed or run disambiguation automatically.

#### L4 - Advanced (observability + evaluation + reliability)
Goal: Add production-grade tooling: evaluation pipelines and observability to measure and
debug system quality.

#### How we decided to address the challenge

Although the challenge is structured as previously described, we have organized our project based on practicality and seamless integration.
In fact, some components of the stack are easily built together so we followed out own path through the predefined levels.

![Election_chat.drawio.png](Election_chat.drawio.png)

The above figure shows the project's architecture based on the requirements.

#### Target machine specs
- Dev/Test: Macbook Pro M4 (24GB)
    - Required Memory: ~16 GB (RAM)
    - Storage: 50-100 GB
- Deployment:
    - [Vast AI](https://vast.ai/) Cloud server equiped with an NVIDIA 4090 GPU (24 GB)
- Tested OS: Unix-based (Ubuntu, Darwin)

PS: The target environment is either GPU- or CPU-based and is not conditioned on the dev environment. 
Runnning the app on CPU was initially meant for debugging and benchmarking purposes, but it happened to be quite sufficient to have a working stack. The only inconvenient would be the latency (up to 10min and more) compared to GPU-based deployment (responses within seconds).

#### Main Stack
- **Docker**: Containerization tool to package the entire stack for consistent deployment across environments.
#### LLM / Agent orchestration
- **vLLM**: High-throughput engine for serving LLMs with optimized memory management.
- **LangChain**: Orchestration framework that links the prompts, retrieval logic, and LLM calls.

#### Data/Metrics Visualization

- **Prometheus**: Time-series database that scrapes and stores metrics from vLLM and the chat app.
- **Grafana**: Visualization platform to view and monitor system health via a dashboard.
- **LangSmith**: for end-to-end Traceability

#### Frontend
- **Streamlit**: Python-based web framework for building a clean chat interface with minimal frontend code.

#### General Backend
- **Python >=3.12**: The primary programming language used for logic, data processing, and glue code.
- **pdfblumber**: Surgical tool for digital PDFs which doesn't "guess" where a character is but asks the PDF file for the exact X/Y coordinates of every letter and line. 

    - We refrain from using vision-language models (VLMs) like Qwen-VL, Dots OCR, DeepSeek OCR or any LM-based OCR model, to avoid unwanted "hallucinations" of the text based on pixels. 
    - In our case where each decimal point/digit matters, `pdfplumber` (or any of its likes) seems more accurate because it extracts the raw data.
    - We also do not want to allocate additional compute resources to run VLMs in addition to our base LLM(s). However, we do acknowledge VLMs as a more generic way of handling various PDF layouts out of the box
    - See [notebooks/pdf_data_extraction.ipynb](./notebooks/pdf_data_extraction.ipynb) for more details about our experiments with the target dataset and how the data extraction pipeline was tailored to the subject matter given we had only one PDF to consider as data source.

### Step 1
Once we found a sufficient design, we started by making sure components are properly implemented within the docker environment with with the appropriate parameters:
- vLLM (foundation)
- Prometheus and Grafana (for observability)
- Streamlit (UI/Access)
- Nginx (reverse proxy)

See [docker-compose.yml](./docker-compose.yml) for more details about how these components are interconnected.

Note that the observability/monitoring aspects were taken care of during the design and initialization of the stack since it allows easy debugging of the system.

#### Setup instructions
Clone the project repo
```bash
$ git clone git@github.com:dric2018/chat-app.git
```

Once the operation is complete, you can then move to the project folder using the command `cd chat-app`.

If you are using a cloud-based instance, make sure to refresh it first. Run the [scripts/update_instance.sh](./scripts/update_instance.sh) script as follows:
```bash
$ chmod +x scripts/update_instance.sh && ./scripts/update_instance.sh
```

Make sure docker is already installed on the host machine. Installation details can be found at [subfuzion/install-docker-ubuntu.md](https://gist.github.com/subfuzion/90e8498a26c206ae393b66804c032b79) on GitHub Gist.

Typically, it can be done by running the following command line:
```sh
$ curl -fsSL https://get.docker.com/ | sh
```

Also install docker-compose (>v5.0.1) as follows:
```sh
$ sudo apt update && sudo apt install docker-compose && sudo apt-get install docker-compose-plugin
```

If Python is not already installed on the machine, you can do so by running the following commands:
```bash
$ sudo apt-get update && sudo apt-get install -y python3.13 python3.13-venv python3.13-dev

# it can be installed on Mac OS X using the `brew install python` command 
# or check the guide from https://docs.python-guide.org/starting/install3/osx/
```

Create a virtual environment (venv) on the host machine
```bash
$ python3.13 -m venv .venv # creating the venv and installing project dependencies
$ source .venv/bin/activate && pip install -e . # then run this to install packages within venv
```

#### DB Creation
We get the duckdb package installed:
```bash
$ snap install && mkdir -p storage/duckdb
```
To create and populate the database, you must run the [notebooks/pdf_data_extraction.ipynb](./notebooks/pdf_data_extraction.ipynb) notebook to generate the required .parquet files (source of truth) and then run the [src/db/election_db.py](./src/db/election_db.py) script as follows:

```bash
(.venv) $ python -m src.db.election_db # which will create an instance of ElectionDB and execute its init_db() procedure
```

We only create the database here. We give more details about it later on.

#### Build stack: 

The LLM orchestration dependencies can be installed by running the `init.py` script as follows:
```bash
(.venv) $ python -m src.init --reset --recreate # run the init script
# this will create the docker containers for running the chat-app
```

The first build may take some time.

Once the Stack is up and running, you will be able to access each service via:

|Service|URL|
|---|---|
|Streamlit UI (via Nginx)| http://localhost:8080/|
|vLLM API   |http://localhost:8000/v1/|
|vLLM metrics   |http://localhost:8000/metrics|
|Grafana (dashboards)	|http://localhost:3000|
|Prometheus	|http://localhost:9090|

PS: if the app is deployed on a remote server, the services will be available on `http://${CFG.SERVER_IP}:${CFG.VLLM_PORT}` as defined in the `.env` file. You may want to check your cloud console for the newly assigned ports as it is usually the case for services like Vast.ai.

The streamlit app may require a username and a password. Use those that you specified in your `.env` file.

Grafana's default login credentials are (admin, admin).

### Step 2: Implementing the Text-to-SQL Agent 

#### Database considerations
For a stable, high-performance workflow involving electoral data, the proposed schema should be split into structural tables (for precise SQL filtering) and vector tables (for semantic search).

Since our data involves multi-line cells and merged cells, using a normalized relational structure is the most reliable way to prevent the "semantic drift" that happens in raw text RAG.

Overall agentic workflow:

User query
    > Intent classifier
    > SQL (output) Generator
        
        # SQL Generator
        > Uses the appropriate tools to gather insights from the db
        > Validates and executes candidate SQL query
        # Generic behavior (SQLAgent, RAGAgent, HybridAgent)
        > Returns:
            - Short narrative
            - Dataframe preview
            - Optional chart (if requested)
            - Not found message if no appropriate answer found

#### DuckDB

I. Relational Schema

Database tables are categorized into:
1. **Dimentions**: Region, Party, Candidate, Constituency, 
2. **Facts**: Turnout (separate), Result (central)


II. Vector Schema 

For retrieval-augmented generation (RAG), we need a "contextual chunk" table. Instead of embedding raw rows from the PDF, we embed human-readable summaries of table rows and store them in our DB for later use. These narratives can be found in the `vw_rag_descriptions` table and the `embeddings` table (which also has the corresponding pre-computed embedding vectors).

Note: Although we did our best to build a RAG-ready pipeline, it should be noted that the restriction on `only using the PDF` as source of truth make it realistically tough to address RAG-enabled queries, e.g. "How did the legislative results impact the formation of the new government in January 2026?" or "How many seats were contested in the National Assembly during this cycle?", which cannot be derived from the PDF data.

III. DB Views

Instead of exposing all raw tables, we expose curated views (see [src/db/views.sql](./src/db/views.sql)). This is a design choice to simplify data access, enhance security, and provide logical data abstraction.

The data extraction notebook generates db-related files under `data/processed`:
- candidates.parquet
- constituencies.parquet
- parties.parquet
- regions.parquet
- results.parquet
- turnout.parquet

#### ElectionDB
DB-related operations are grouped into the `ElectionDB` class in [src/db/election_db.py](./src/db/election_db.py). Those are:
- `init_db()` for data base initialization
- `deploy_views()` to create and deploy table views
- `load_embedding_model()` to load the specified embedding model
- `compute_embeddings()` to compute embeddings and insert them into the database
- Search 
    - `vector_search()` (VS) for performing vector search on the stored data
    - `full_text_search()` (FTS) for performing vector search on the stored data
    - `hybrid_search()` for combining both VS and FTS

#### Agent Implementation
See [src/agent.py](./src/agent.py) for further details about the design of the Agents.

- `SQLAgent`: generates SQL queries from the user prompt
- `RAGAgent`: Interacts with the database to answer questions about explanations, and grounding.
- `HybridAgent`: routes the user request based on the identified intent and path.

#### Base LLMs
CPU (thinking disabled):
- Qwen/Qwen3-0.6B ( ✅ ): tiny and fast execution with tool-calling enabled
    - set MAX_TOKENS = 1024
- Qwen/Qwen3-1.7B ( ✅ ): tiny and fast, but relatively bigger than previous version. This version and the previous one require less than 16GB of memory to run.
    - set MAX_TOKENS = 1024
- Qwen/Qwen3-4B-Instruct-2507 ( ✅ ):  small model, much slower but more robust and more reliable with tool manipulation as well as text generation.
    - set MAX_TOKENS = 4096
- facebook/opt-125m ( ❌ ): could not make it work with the setup

GPU (thinking disabled):
- Qwen/Qwen3-4B-Instruct-2507 ( ✅ ):  small model, much slower but more robust and more reliable with tool manipulation as well as text generation.
    - set MAX_TOKENS = 4096

### Overall Progress
Level 1: Analytics-First Agent (95% Complete)
- Ingestion: ✅ 

>A reproducible DuckDB pipeline with normalized entities and relational joins was presented in Setp 2. See [src/db/election_db.py](./src/db/election_db.py) and [src/db/views.sql](./src/db/views.sql) for additional details.
- SQL Agent: ✅ 
> Intent classification ✅

> Restricted SQL generation with security guardrails; no forbidden statement is executed and violations are blocked ✅.

> Chart Generation: We have the `CHART` intent and the agent returns the necessary data; the streamlit UI displays the charts via plotly, though it can be improved ✅. Currently, only pie, bar charts, and histograms are supported.

Level 2: Hybrid Router (90% Complete)
RAG Indexing: ✅
> Created the `embeddings` table and its corresponding view `vw_rag_descriptions` for results and turnout.

Hybrid Routing: ✅. 
> Logic for SQL and RAG merged into a `HybridAgent` that uses the appropriate pipeline based on the user request (intent, path).

> A CHAT route was added in case the user asks general questions that may not be directly related to the elections. 

> DuckDB has native support for string similarity functions like `levenshtein`, `hamming`, and `jaro_winkler_similarity`. Thus we mainly rely on these for spellcheking. A special `entity_alias` table was created to allow for initial correction of the user prompt when necessary, matching names that are similar to known `constituency` names.

Citations: 🏗️ Not yet implemented. However, we extracted the page_id (source page number) during the ingestion process. 

> `ENTITY_ID` present in the RAG-ready tables; 
> Need to map it back to `page_id` during the final response, after we ensure `page_id` and `row_id` are extracted and saved during ingestion.

Level 3: Improved Agentic (60% Complete)
Disambiguation: 🏗️ In Progress. 

> Current ingestion handles some normalization, but the Agent doesn't yet ask the user for clarification in a controlled loop due to untracked chat history outside of the sessions.

Session Memory: 🏗️ Not yet implemented. 

> This is currently handled using `st.session_state` in the Streamlit frontend. We did not spend time adding persistent memory, but it should be noted that more advanced UI like `Open WebUI` offer this for free.

Level 4: Observability & Evaluation (60% Complete)

Observability: ✅ Started. 

> Centralized logger and a metrics dictionary in the `Agent` class. ✅

> Grafana dashboard to visualize LLM metrics ✅

Available metircs include:
- Final response latency
- Token usage/generated
- KV cache % (GPU & CPU)
- Total Requests Completed
- Total Tokens (Gen / Prompt)
- Avg Tokens per Request
- User Perceived Latency (P95)
- System Throughput (Tokens/s)
- Workload Distribution (Running vs Waiting)
- vLLM CPU Usage
- Cache Hit Rate (Prefix Caching)

Traceability: 🏗️ Added LangSmith to the stack. 

> This is combined with the Grafana dashboard and allows for tracking:
- Intent classification \& routing
- Retrieval results
- SQL generated and validation outcome
tool calls (charts) and timings
- Blocked Security Violations

Evaluation Suite: 🏗️ Not yet implemented. 

> Still need an offline script to measure "Fact lookup accuracy" and "Citation faithfulness".

#### Typical Workflow: The "Ask-Route-Execute" Loop

- User Input: User asks, "Who got the most votes in Abidjan?"
- Intent Classification (_get_intent):
    - The HybridAgent uses the LLM to categorize this.
    - Result: QueryIntent.RANKING.
    
- Routing:
    - The HybridAgent sees RANKING and internally delegates the task to its SQLAgent instance.

- Specialist Processing (process_query):
    - The SQLAgent takes over.
    - It retrieves the Schema Context (Tool Call 1/describe_db) and then continues with any necessary tool calls until sufficient insights are gathered.
    - It generates a query using appropriate statements.
    - It runs `validate_sql` to ensure SQL query does not violate security principles.

- Data Retrieval:
    - The SQL is executed against DuckDB in read_only mode.
    - Result: A DataFrame containing the candidate and vote count.

- Interpretation (_interpret_results):
    - The raw DataFrame is turned into a natural sentence: "In Abidjan, Candidate X leads with Y votes."

- Final Response: The HybridAgent returns a structured dictionary containing the final answer and the raw data for a frontend table and/or chart generation.

### Deliverables (all levels)
- Source code repo with:
    - ingestion pipeline ✅ (See init_db() in for data `src/db/election_db.py`)
    - app ✅
    - tests (as relevant per level)
- README + .env.example ✅

- Short write-up ([cmanouan_solution_chat_app.pdf](./cmanouan_solution_chat_app.pdf)):
    - Video capture of the solution ✅
    - Description of the work done ✅
    - schema decisions ✅
    - routing/guardrails (if implemented) ✅; See [src/db/sql_agent.py](./src/db/sql_agent.py)
    - known limitations + next steps ✅

To be added/Future work:
- Persistent messages history
- Refined Monitoring dashborad
- Move prompts from within functions/methods to a centralized registory for the agents to consume them seamlessly
- Add support for French prompts/queries

# Credits and Acknowledgement
- The project was implemented with partial assistance from LLMs (Qwen3-8b, Gemini-3-12b, Mistral-22b).
- No typical coding agent was used in this project.