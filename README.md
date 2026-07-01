# SHL Assessment Recommender

Conversational agent that recommends SHL **Individual Test Solutions** through dialogue.
FastAPI service with `GET /health` and `POST /chat` per the assignment spec.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your Gemini API key (https://aistudio.google.com/apikey)
```

## 2. Run locally

```bash
export $(cat .env | xargs)   # or use python-dotenv / your shell's env loading
uvicorn app.main:app --reload --port 8000
```

Check it's alive:

```bash
curl http://localhost:8000/health
```

Try a conversation:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## 3. Run the local eval

Official traces weren't available to us at build time (see `APPROACH.md`), so
`tests/traces.json` has 9 synthetic scripted conversations covering clarify,
recommend, refine, compare, refuse (legal/general-advice/prompt-injection),
and the turn cap. With the server running:

```bash
python tests/run_eval.py
```

## 4. Deploy (free tier)

**Render (recommended, `render.yaml` included):**
1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → New → Blueprint → point at the repo.
3. It reads `render.yaml`, builds the Dockerfile, and asks for `GEMINI_API_KEY` — paste it.
4. Wait for deploy; first `/health` call after idle can take up to ~2 min (free-tier cold start, matches the assignment's grace period).

**Railway / Fly / Modal:** the included `Dockerfile` works as-is — create a new
service from the repo, set `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) as
an environment variable, deploy.

## 5. Re-scraping the catalog (optional)

`data/shl_catalog.json` (258 Individual Test Solutions) was built from a
public pre-scraped CSV (see `APPROACH.md` for why — SHL's catalog listing page
did not return a scrapable listing when this was built) filtered down to
single-test-type items and de-duplicated. `scripts/rebuild_catalog.py`
documents/reproduces that cleaning step from a raw CSV of
`name,url,duration,test_type,remote_testing,adaptive_irt` if you have a fresher
scrape to feed it.

## Project layout

```
app/
  main.py       FastAPI app, /health and /chat
  agent.py      Orchestration: retrieval, prompt building, guardrails, validation
  catalog.py    Catalog loading + TF-IDF retrieval + fuzzy compare lookup
  llm.py        Gemini REST wrapper (structured JSON output)
  schemas.py    Pydantic request/response models
data/
  shl_catalog.json   258 Individual Test Solutions
tests/
  traces.json    synthetic conversation traces
  run_eval.py    local eval harness
scripts/
  rebuild_catalog.py   reproducible cleaning step
```
