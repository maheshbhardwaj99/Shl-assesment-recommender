import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent
from .catalog import get_catalog
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_agent")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request, so /chat calls stay inside the 30s budget.
_catalog = get_catalog()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = run_agent(req.messages, _catalog)
    except Exception:  # noqa: BLE001 - never let an unhandled error break the schema contract
        logger.exception("agent failure")
        result = {
            "reply": "Sorry, something went wrong on my end. Could you rephrase your request?",
            "recommendations": [],
            "end_of_conversation": len(req.messages) + 1 >= 8,
        }
    return ChatResponse(**result)


@app.get("/")
def root():
    return {
        "service": "SHL Assessment Recommender",
        "catalog_size": _catalog.size(),
        "endpoints": ["/health", "/chat"],
    }
