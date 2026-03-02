"""FastAPI service for original Qwen model inference."""

import os
from pathlib import Path

import torch
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Embedded default prompt for infer when none provided
DEFAULT_PROMPT = os.environ.get(
    "DEFAULT_PROMPT", "What is 2 + 2? Reply in one short sentence."
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_PATH = os.environ.get("MODEL_PATH", "")

app = FastAPI(title="Original Qwen Inference Service")
_model = None
_tokenizer = None


def _device() -> torch.device:
    use_cuda = os.environ.get("DEVICE", "cpu").lower() == "cuda"
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    device = _device()
    path = MODEL_PATH.strip()
    model_id = path if path and Path(path).is_dir() else MODEL_ID
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )


class InferRequest(BaseModel):
    """Request body for /infer."""

    prompt: str | None = Field(
        default=None, description="User prompt; if omitted, default prompt is used."
    )


class InferResponse(BaseModel):
    """Response for /infer."""

    reply: str
    prompt_used: str


@app.get("/health")
def health():
    """Readiness check."""
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest | None = Body(None)):
    """Run inference; uses embedded default prompt if prompt not provided."""
    try:
        _load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model load failed: {e!s}")
    prompt = (req.prompt if req and req.prompt is not None else None) or DEFAULT_PROMPT
    messages = [{"role": "user", "content": prompt}]
    try:
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat template failed: {e!s}")
    inputs = _tokenizer(text, return_tensors="pt")
    device = _device()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e!s}")
    input_len = inputs["input_ids"].shape[1]
    reply_ids = outputs[0][input_len:]
    if isinstance(reply_ids, torch.Tensor) and reply_ids.is_cuda:
        reply_ids = reply_ids.cpu()
    reply = _tokenizer.decode(reply_ids, skip_special_tokens=True)
    return InferResponse(reply=reply.strip(), prompt_used=prompt)


@app.on_event("startup")
def startup():
    """Optional: preload model on startup (can be disabled via env)."""
    if os.environ.get("PRELOAD_MODEL", "0") == "1":
        _load_model()
