# Hybrid Buyer Advisor - Backend

A multi-agent AI assistant for real estate property search, using **Superlinked** for semantic vector search with **Qdrant Cloud** as the persistent vector database.

## Features

- **Multi-modal Search**: Combines semantic text search with numeric similarity (price, bedrooms, size)
- **Natural Language Queries**: Ask questions like "3 bedroom house under $500k in Austin"
- **Persistent Storage**: Data stored in Qdrant Cloud (survives restarts)
- **Multi-Agent Architecture**: Specialized agents for search, valuation, comparison, and more

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Hybrid Buyer Advisor                        │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application (main.py)                                   │
│    ├── Query Endpoint                                            │
│    ├── Favorites Management                                      │
│    └── Health Check                                              │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Agent System (LangGraph)                                  │
│    ├── Router Agent         → Intent classification              │
│    ├── Filter Agent         → Property search                    │
│    ├── Valuation Agent      → Price estimation                   │
│    ├── Comparison Agent     → Trade-off analysis                 │
│    ├── Market Insights      → Area statistics                    │
│    └── Favorites Agent      → Manage saved properties            │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                            │
│    └── Superlinked PropertySearchService                         │
│          ├── Schema: Property (price, bed, bath, city, etc.)     │
│          ├── Spaces: description, price, bed, bath, size, lot    │
│          ├── Query: Natural language + filters                   │
│          └── Backend: Qdrant Cloud                               │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.11 or 3.12
- Qdrant Cloud account (free tier available at https://cloud.qdrant.io)
- OpenAI API key

### 2. Setup Environment

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```env
# OpenAI (required)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Qdrant Cloud (required)
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

---

## Data Ingestion

### Step 1: Prepare Your Dataset

Place your dataset at `data/realtor-data.csv`. The CSV should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `price` | float | Listing price in USD |
| `bed` | int | Number of bedrooms |
| `bath` | float | Number of bathrooms |
| `house_size` | float | Square footage |
| `acre_lot` | float | Lot size in acres |
| `city` | string | City name |
| `state` | string | State name |
| `status` | string | "for_sale" or "sold" |
| `zip_code` | string | ZIP code |
| `street` | string | Street identifier |
| `brokered_by` | string | Broker ID |
| `prev_sold_date` | string | Previous sale date |

### Step 2: Test with a Small Sample

Always test with a small sample first to verify everything works:

```bash
python -m scripts.ingest_properties \
    --data-path ./data/realtor-data.csv \
    --sample-size 1000 \
    --batch-size 500 \
    --verify
```

### Step 3: Ingest Full Dataset

Once the test passes, ingest the full dataset:

```bash
python -m scripts.ingest_properties \
    --data-path ./data/realtor-data.csv \
    --batch-size 500 \
    --verify
```

**Ingestion Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Path to CSV file | Required |
| `--sample-size` | Number of records to sample | All records |
| `--batch-size` | Records per batch (reduce if hitting size limits) | 500 |
| `--verify` | Run test queries after ingestion | False |

**Expected Output:**
```
==================================================
PROPERTY DATA INGESTION
==================================================
Data path: ./data/realtor-data.csv
Sample size: All records
Batch size: 500
Qdrant URL: https://your-cluster.cloud.qdrant.io
==================================================

Initializing PropertySearchService...
PropertySearchService initialized with Qdrant RestExecutor
Loading data from ./data/realtor-data.csv
Loaded 1037588 records

Ingesting data in batches...
Ingesting batches: 100%|████████████████| 2076/2076 [45:32<00:00, 1.32s/it]

✅ Successfully ingested 1037588 properties
✅ Done!
```

---

## Monitoring Qdrant

### Check Point Count (CLI)

Monitor how many records have been ingested:

```bash
# Replace with your Qdrant URL and API key
curl -s "https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333/collections/default" \
  -H "api-key: YOUR_QDRANT_API_KEY" | python -c "
import sys, json
data = json.load(sys.stdin)
result = data.get('result', {})
print(f\"Points: {result.get('points_count', 0):,}\")
print(f\"Indexed: {result.get('indexed_vectors_count', 0):,}\")
print(f\"Status: {result.get('status', 'unknown')}\")
"
```

### Real-time Monitoring

Watch the point count update in real-time during ingestion:

```bash
# Run in a separate terminal while ingestion is running
while true; do
  count=$(curl -s "https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333/collections/default" \
    -H "api-key: YOUR_QDRANT_API_KEY" | grep -o '"points_count":[0-9]*' | cut -d: -f2)
  echo "$(date '+%H:%M:%S') - Points: $count"
  sleep 10
done
```

### Qdrant Cloud Dashboard

1. Go to https://cloud.qdrant.io
2. Sign in to your account
3. Click on your cluster
4. Navigate to **Collections** tab
5. View the `default` collection to see:
   - **Points count**: Number of ingested records
   - **Vectors count**: Number of vectors stored
   - **Collection status**: Health status

### List All Collections

```bash
curl "https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333/collections" \
  -H "api-key: YOUR_QDRANT_API_KEY"
```

### Delete Collection (Reset)

If you need to start fresh:

```bash
curl -X DELETE "https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333/collections/default" \
  -H "api-key: YOUR_QDRANT_API_KEY"
```

---

## Run the API Server

```bash
python -m app.main
# Or: uvicorn app.main:app --reload --port 8000
```

API available at: http://localhost:8000

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Settings & environment
│   ├── agents/                 # Multi-agent system
│   │   ├── base_agent.py
│   │   ├── router_agent.py
│   │   ├── filter_agent.py
│   │   ├── valuation_agent.py
│   │   ├── comparison_agent.py
│   │   ├── market_insights_agent.py
│   │   ├── favorites_agent.py
│   │   └── conversation_manager.py
│   ├── infrastructure/         # External service integrations
│   │   └── superlinked/
│   │       ├── __init__.py
│   │       ├── constants.py    # Min/max values, state list
│   │       ├── index.py        # Schema & embedding spaces
│   │       ├── query.py        # Query definition
│   │       └── service.py      # PropertySearchService
│   ├── models/
│   │   ├── schemas.py          # Pydantic models
│   │   └── state.py            # LangGraph state
│   ├── services/
│   │   ├── llm_service.py      # OpenAI integration
│   │   └── session_service.py  # User sessions
│   └── workflow/
│       └── graph.py            # LangGraph workflow
├── scripts/
│   ├── ingest_properties.py    # Data ingestion script
│   └── create_sample_data.py   # Generate test data
├── data/
│   └── realtor-data.csv        # Property dataset
├── System_Prompts/             # Agent prompt templates
├── requirements.txt
├── env.example
└── README.md
```

---

## Superlinked Integration

### How It Works

1. **Schema**: Defines property fields (price, bed, bath, etc.)
2. **Embedding Spaces**: Creates vector spaces for semantic + numeric similarity
3. **Index**: Combines all spaces for multi-modal search
4. **Query**: Supports natural language with LLM-powered parameter extraction
5. **Qdrant Backend**: Persists vectors and enables fast similarity search

### Search Features

**Natural Language Queries:**
```python
results = await service.search_properties(
    "3 bedroom house under $400000 in California",
    limit=10
)
```

**With Explicit Filters:**
```python
results = await service.search_properties(
    "spacious family home",
    limit=10,
    state="Texas",
    min_bed=3,
    max_price=500000
)
```

### Embedding Spaces

| Space | Field | Mode | Description |
|-------|-------|------|-------------|
| `description_space` | description | Text Similarity | Semantic search on property descriptions |
| `price_space` | price | MINIMUM | Lower prices score higher |
| `bed_space` | bed | MAXIMUM | More bedrooms score higher |
| `bath_space` | bath | MAXIMUM | More bathrooms score higher |
| `house_size_space` | house_size | MAXIMUM | Larger homes score higher |
| `acre_lot_space` | acre_lot | MAXIMUM | Larger lots score higher |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Natural language property search |
| `/favorites` | GET | Get user's saved properties |
| `/favorites` | POST | Add property to favorites |
| `/favorites/{id}` | DELETE | Remove from favorites |

---

## Troubleshooting

### "Connection to Qdrant failed"

1. Check `QDRANT_URL` includes `https://`
2. Verify `QDRANT_API_KEY` is correct
3. Ensure Qdrant cluster is running (check dashboard)

### "Falling back to InMemoryExecutor"

This means Qdrant connection failed. Common causes:
1. **Collection mismatch**: Delete the existing collection and re-run ingestion
2. **Invalid credentials**: Check your API key
3. **Network issues**: Verify you can reach the Qdrant URL

```bash
# Delete existing collection
curl -X DELETE "https://YOUR-CLUSTER.cloud.qdrant.io:6333/collections/default" \
  -H "api-key: YOUR_API_KEY"

# Re-run ingestion
python -m scripts.ingest_properties --data-path ./data/realtor-data.csv --batch-size 500
```

### "Payload too large" Error

Reduce batch size:
```bash
python -m scripts.ingest_properties \
    --data-path ./data/realtor-data.csv \
    --batch-size 200
```

### "No results found"

1. Verify data was ingested: check Qdrant dashboard for collection point count
2. Try broader queries without filters
3. Check if dataset was sampled too small

### "Superlinked import error"

```bash
pip install --upgrade "superlinked>=37.5.0"
```

---

## Development

```bash
# Run tests
pytest

# Format code
black app/ scripts/
isort app/ scripts/

# Type check
mypy app/
```

---

## License

MIT
