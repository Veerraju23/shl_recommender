# SHL Assessment Recommender

A conversational FastAPI agent that guides hiring managers from a vague role description to a
grounded shortlist of SHL Individual Test Solution assessments.

## Quick Start (Local)

```bash
# 1. Clone / enter the project
cd shl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY=sk-ant-...

# 4. Start the server
uvicorn main:app --reload --port 8000
```

The service is now live at http://localhost:8000

- **Health check**: `GET /health` → `{"status": "ok"}`
- **Interactive docs**: http://localhost:8000/docs

---

## API Reference

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200. Used by the evaluator to check readiness.

---

### `POST /chat`

**Request body:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure! What seniority level are you targeting?"},
    {"role": "user", "content": "Mid-level, around 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java developer with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Notes:**
- `recommendations` is `[]` when the agent is still clarifying or refusing
- `end_of_conversation` is `true` only when the task is complete
- Max 8 turns per conversation (evaluator limit)

---

## Example Conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need an assessment for a software engineer role."}
    ]
  }'
```

Expected: agent asks a clarifying question (seniority, specific tech stack).

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need an assessment for a software engineer role."},
      {"role": "assistant", "content": "Happy to help! What seniority level are you targeting?"},
      {"role": "user", "content": "Mid-level Python developer, will work on data pipelines."}
    ]
  }'
```

Expected: recommendations including Python knowledge test, possibly SQL and data analysis.

---

## Running Tests

```bash
# Unit + integration tests (no API key needed)
pytest tests.py -v -k "not Behavior"

# All tests including live model probes (requires ANTHROPIC_API_KEY)
pytest tests.py -v
```

---

## Deployment on Render (Free Tier)

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo and set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `ANTHROPIC_API_KEY = your_key_here`
5. Deploy — your URL will be `https://your-app.onrender.com`

The `/health` endpoint allows up to 2 minutes for cold-start wake-up on free tier.

---

## Project Structure

```
shl_recommender/
├── main.py              # FastAPI app, system prompt, response parsing
├── catalog.json         # SHL Individual Test Solutions catalog (67 products)
├── tests.py             # Test suite (26 unit tests + live behavior probes)
├── requirements.txt     # Python dependencies
├── Dockerfile           # For containerized deployment
├── render.yaml          # Render.com deployment config
├── .env.example         # Environment variable template
└── approach_document.md # Design decisions (2-page summary)
```

---

## Architecture Notes

- **Catalog in-context**: All 67 assessments are embedded in the system prompt as structured text.
  No vector database needed at this catalog size.
- **Hallucination guard**: Every recommendation's URL is validated against the catalog before
  returning. Hallucinated URLs are silently dropped; known product names with wrong URLs are
  corrected automatically.
- **Stateless**: No session storage. Full conversation history sent on every `/chat` call.
- **Timeout-safe**: Anthropic API call uses a 28s timeout (within the evaluator's 30s limit).
