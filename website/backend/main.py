"""
FastAPI backend for the prompt-compress demo site.

Wraps the installed `prompt_compress` library and exposes a single endpoint
for live compression. Designed for local-only use during the presentation —
no auth, no rate limiting, no persistence.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prompt_compress import PromptCompressor

app = FastAPI()

# CORS is generous because we don't know which port Vite will pick if 5173
# is taken — and this server only ever runs locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singleton: SentenceTransformer takes ~3s to load. We want that
# cost paid at startup, not on the first user click during the demo.
compressor = PromptCompressor()
_warmup = compressor.compress(
    "You are a helpful assistant. Answer clearly and avoid jargon."
)
print(
    f"Backend ready. Warmup compression: {_warmup.compression_ratio:.1%} "
    f"(tier {_warmup.tier}, {_warmup.time_seconds:.2f}s)"
)


class CompressRequest(BaseModel):
    prompt: str
    min_similarity: float = 0.75


@app.post("/api/compress")
def compress(req: CompressRequest):
    try:
        result = compressor.compress(
            req.prompt,
            min_similarity=req.min_similarity,
            on_failure="fallback",
        )
        return {"ok": True, "result": result.to_dict()}
    except Exception as e:
        # Any exception means the pipeline itself broke; surface the message
        # to the client so the demo doesn't show a generic 500.
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/health")
def health():
    return {"ok": True}
