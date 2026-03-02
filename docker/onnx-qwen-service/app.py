"""FastAPI service for ONNX-exported Qwen model inference."""

import os
import sys
from pathlib import Path

import torch
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Embedded default prompt when none provided
DEFAULT_PROMPT = os.environ.get(
    "DEFAULT_PROMPT", "What is 2 + 2? Reply in one short sentence."
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "/model")

app = FastAPI(title="ONNX Qwen Inference Service")
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    path = Path(ONNX_MODEL_PATH).resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"ONNX model dir not found: {path}. "
            "Mount your ONNX export (e.g. from qwen-export.py) with: "
            "docker run -v /path/to/qwen25_05b_onnx:/model ..."
        )
    device = os.environ.get("DEVICE", "cpu").lower()
    try:
        if device == "cuda":
            try:
                _model = ORTModelForCausalLM.from_pretrained(
                    str(path),
                    export=False,
                    provider="CUDAExecutionProvider",
                )
            except (ValueError, Exception):
                _model = ORTModelForCausalLM.from_pretrained(
                    str(path),
                    export=False,
                    provider="CPUExecutionProvider",
                )
        else:
            _model = ORTModelForCausalLM.from_pretrained(
                str(path),
                export=False,
                provider="CPUExecutionProvider",
            )
    except Exception as e:
        if "InvalidProtobuf" in type(e).__name__ or "INVALID_PROTOBUF" in str(e):
            print(
                "ONNX model failed to load (InvalidProtobuf). Re-export without OnnxSlim.",
                file=sys.stderr,
            )
        raise
    _tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
        fix_mistral_regex=True,
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
    _load_model()
    prompt = (req.prompt if req and req.prompt is not None else None) or DEFAULT_PROMPT
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(text, return_tensors="pt")
    device = getattr(_model, "device", torch.device("cpu"))
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = _model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    reply_ids = outputs[0][input_len:]
    if isinstance(reply_ids, torch.Tensor) and reply_ids.is_cuda:
        reply_ids = reply_ids.cpu()
    reply = _tokenizer.decode(reply_ids, skip_special_tokens=True)
    return InferResponse(reply=reply.strip(), prompt_used=prompt)


@app.on_event("startup")
def startup():
    """Optional: preload model on startup."""
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        try:
            _load_model()
        except FileNotFoundError:
            pass
