"""Embedding + reranking service.

Holds two small models resident on the GPU:
  BAAI/bge-m3               (568M, 1024-dim, 100+ languages)  ~1.2 GB in fp16
  BAAI/bge-reranker-v2-m3   (568M cross-encoder)              ~1.2 GB in fp16

Both are Apache/MIT licensed and run comfortably alongside an 8B LLM on 12 GB.
"""
import os
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="rag-embeddings")
_lock = threading.Lock()  # models are not thread-safe; serialise GPU access
_state: dict = {}


@app.on_event("startup")
def load_models() -> None:
    print(f"[embeddings] device={DEVICE} torch={torch.__version__}", flush=True)
    if DEVICE == "cuda":
        print(f"[embeddings] gpu={torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}", flush=True)
    _state["embedder"] = SentenceTransformer(
        EMBED_MODEL, device=DEVICE, model_kwargs={"torch_dtype": DTYPE}
    )
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=1024)
    if DEVICE == "cuda":
        reranker.model.half()   # CrossEncoder has no model_kwargs; convert after load
    _state["reranker"] = reranker
    print("[embeddings] models loaded", flush=True)


class EmbedRequest(BaseModel):
    texts: list[str]
    # bge-m3 needs no instruction prefix for either side — one of the reasons it is
    # a convenient default. Kept as a knob in case you swap in an instructed model.
    prefix: str = ""
    batch_size: int = 16


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = Field(default=6, ge=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "embedder" in _state else "loading", "device": DEVICE}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    texts = [req.prefix + t for t in req.texts]
    with _lock:
        vecs = _state["embedder"].encode(
            texts,
            batch_size=req.batch_size,
            normalize_embeddings=True,   # required: schema uses cosine distance
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return {"embeddings": [v.tolist() for v in vecs], "dim": int(vecs.shape[1])}


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    if not req.documents:
        return {"results": []}
    pairs = [(req.query, d) for d in req.documents]
    with _lock:
        scores = _state["reranker"].predict(pairs, batch_size=8, show_progress_bar=False)
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
    return {
        "results": [
            {"index": int(i), "score": float(s)} for i, s in ranked[: req.top_k]
        ]
    }
