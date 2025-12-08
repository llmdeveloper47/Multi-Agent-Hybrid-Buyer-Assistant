# Hybrid Buyer Advisor - Backend

A multi-agent AI assistant for real estate buyers, powered by **LangGraph**, **Superlinked**, and **OpenAI GPT** models.

## Features

- **Natural Language Property Search**: Ask questions like "Find 3-bedroom houses under $500k in Seattle"
- **Personalized Recommendations**: Get property suggestions based on your preferences
- **Price Valuation Estimates**: Understand if a property is fairly priced
- **Trade-off Analysis**: Compare properties and understand pros/cons
- **Favorites Management**: Save and manage your favorite properties

## Architecture

### Superlinked + Qdrant Integration

The system uses **[Superlinked](https://docs.superlinked.com/)** with **Qdrant** as the vector database backend:

```
User Query → LangGraph Agents → Superlinked (Query Engine) → Qdrant (Vector DB) → Results
```

**Superlinked provides:**
- **Multi-modal vector search** - Combines text similarity with numeric filtering (price, beds, sqft)
- **Natural language query understanding** - LLM-powered query interpretation via OpenAI
- **Dynamic parameter weighting** - Adjust search weights at query time
- **Automatic embedding** - Uses state-of-the-art models (Alibaba-NLP/gte-large-en-v1.5)

**Qdrant provides:**
- **Persistent vector storage** - Data survives restarts
- **High-performance search** - Optimized for similarity queries
- **Metadata filtering** - Filter by price, location, bedrooms, etc.

### Multi-Agent Workflow (LangGraph)

```
                         ┌──────────────┐
                         │    Router    │
                         │   (Intent)   │
                         └──────┬───────┘
                                │
         ┌──────────┬───────────┼───────────┬──────────┐
         ▼          ▼           ▼           ▼          ▼
   ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
   │  Filter  ││Valuation ││Comparison││  Market  ││ Favorites│
   │  Agent   ││  Agent   ││  Agent   ││ Insights ││  Agent   │
   └────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
        │           │           │           │           │
        └───────────┴───────────┼───────────┴───────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │   Conversation   │
                     │     Manager      │
                     └──────────────────┘
```

### Agents

| Agent | Purpose |
|-------|---------|
| **Router** | Classifies user intent and routes to specialist agents |
| **Filter & Matching** | Property search using Superlinked natural language queries |
| **Valuation** | Property value estimates and pricing analysis |
| **Comparison** | Trade-off analysis and property comparisons |
| **Market Insights** | Real estate market statistics and trends |
| **Favorites** | Manages user's saved properties |
| **Conversation Manager** | Orchestrates dialogue and composes final responses |

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration settings
│   ├── models/
│   │   ├── schemas.py             # Pydantic API models
│   │   └── state.py               # LangGraph state definitions
│   ├── agents/
│   │   ├── base_agent.py          # Base agent with Superlinked access
│   │   ├── router_agent.py        # Intent classification
│   │   ├── filter_agent.py        # Property search (uses Superlinked)
│   │   ├── valuation_agent.py     # Price analysis
│   │   ├── comparison_agent.py    # Property comparisons
│   │   ├── market_insights_agent.py
│   │   ├── favorites_agent.py
│   │   └── conversation_manager.py
│   ├── services/
│   │   ├── superlinked_service.py # Superlinked + Qdrant integration
│   │   ├── llm_service.py         # OpenAI GPT integration
│   │   └── session_service.py     # User session management
│   ├── workflow/
│   │   └── graph.py               # LangGraph workflow definition
│   └── utils/
│       └── helpers.py
├── System_Prompts/                # Editable agent prompts
│   ├── router_agent_prompt.txt
│   ├── filter_agent_prompt.txt
│   ├── valuation_agent_prompt.txt
│   ├── comparison_agent_prompt.txt
│   ├── market_insights_agent_prompt.txt
│   ├── favorites_agent_prompt.txt
│   └── conversation_manager_prompt.txt
├── scripts/
│   ├── data_ingestion_superlinked.py  # Ingest data into Superlinked/Qdrant
│   └── create_sample_data.py          # Generate test data
├── data/                          # Dataset directory
├── requirements.txt
├── env.example
├── run.py                         # Quick start script
└── README.md
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- OpenAI API key

### 2. Installation

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-api-key-here
```

### 4. Start Qdrant (Vector Database)

**Option A: Docker (Recommended)**
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

**Option B: Qdrant Cloud**
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster and get your API key
3. Update `.env`:
   ```
   QDRANT_URL=https://your-cluster.cloud.qdrant.io
   QDRANT_API_KEY=your-api-key
   ```

**Verify Qdrant is running:**
```bash
curl http://localhost:6333/collections
```

### 5. Prepare & Ingest Data

The system uses the **realtor-data.csv** dataset:

```bash
# Ingest data into Superlinked/Qdrant (sample of 10,000 records)
python -m scripts.data_ingestion_superlinked \
    --data-path ./data/realtor-data.csv \
    --sample-size 10000 \
    --verify

# For larger datasets
python -m scripts.data_ingestion_superlinked \
    --data-path ./data/realtor-data.csv \
    --sample-size 50000 \
    --verify
```

**Dataset Schema:**
```
brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date
```

### 6. Run the Server

```bash
# Quick start
python run.py

# Or with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Query the Assistant

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "message": "Find 3 bedroom houses in Florida under $300k"
  }'
```

### Example Queries

| Query Type | Example |
|------------|---------|
| **Search** | "Find 4 bed homes in Texas around $500k" |
| **Valuation** | "Is $400k a good price for a 3 bed house in Miami?" |
| **Comparison** | "Compare the first two properties" |
| **Market** | "What's the average home price in California?" |
| **Favorites** | "Add this to my favorites" |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check (Superlinked + Qdrant status) |
| POST | `/query` | Send a message to the assistant |
| GET | `/favorites/{session_id}` | Get user's favorites |
| POST | `/favorites` | Add property to favorites |
| DELETE | `/favorites` | Remove from favorites |
| GET | `/stats` | System statistics |

### Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI.

## System Prompts

Agent prompts are stored in `System_Prompts/` as editable text files:

| File | Agent | Purpose |
|------|-------|---------|
| `router_agent_prompt.txt` | Router | Intent classification |
| `filter_agent_prompt.txt` | Filter | Property search formatting |
| `valuation_agent_prompt.txt` | Valuation | Price analysis |
| `comparison_agent_prompt.txt` | Comparison | Property comparisons |
| `market_insights_agent_prompt.txt` | Market | Market statistics |
| `favorites_agent_prompt.txt` | Favorites | Favorites management |
| `conversation_manager_prompt.txt` | Manager | Response composition |

**Changes take effect immediately** (no restart needed).

## Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | **Required** |
| `OPENAI_MODEL` | GPT model | `gpt-4` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `QDRANT_URL` | Full Qdrant URL (overrides host:port) | - |
| `QDRANT_API_KEY` | Qdrant Cloud API key | - |
| `SEARCH_TOP_K` | Number of search results | `10` |

## Troubleshooting

### "Superlinked not connected"

1. Ensure Qdrant is running: `curl http://localhost:6333/collections`
2. Check Docker: `docker ps`
3. Verify `.env` settings

### "OpenAI API Error"

1. Check `OPENAI_API_KEY` in `.env`
2. Verify account has credits
3. Try `gpt-3.5-turbo` for lower cost

### Data Ingestion Issues

```bash
# Verify data file exists
ls -la ./data/realtor-data.csv

# Run with verbose output
python -m scripts.data_ingestion_superlinked \
    --data-path ./data/realtor-data.csv \
    --sample-size 1000 \
    --verify
```

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black app/
isort app/

# Type checking
mypy app/
```

## License

MIT License
