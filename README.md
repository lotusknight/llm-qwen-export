# qwen-export.py — API spec for backend prototype

This document describes **inputs**, **outputs**, and **important flags** of `qwen-export.py` for use as a prototype when implementing a backend service that: pulls a **remote model tar from MinIO**, extracts it, then runs the same export pipeline.

---

## Purpose

- **Script:** Exports a Qwen-style causal LM (e.g. Qwen2.5) from PyTorch/Transformers weights to ONNX, with optional OnnxSlim optimization.
- **Backend use:** Replace the current “model_id → ModelScope download” step with “MinIO tar URL/path → download & extract → same export”. All other inputs and the output format stay the same.

---

## Input fields

| Field | CLI | Type | Required | Default | Description |
|-------|-----|------|----------|---------|-------------|
| **model_id** | positional (or `-` for default) | string | No | `Qwen/Qwen2.5-0.5B-Instruct` | **In script:** ModelScope model ID; weights are resolved to `{cache_root}/{safe_name}` where `safe_name = model_id.replace("/", "--")`. **In backend:** Replace with path to **extracted model dir** (after pulling tar from MinIO and untarring), or a single identifier that your server maps to that path. |
| **output** | `-o`, `--output` | string | No | `./qwen25_05b_onnx` | Directory where the ONNX model and tokenizer will be written. Must be writable; created if missing. |
| **device** | `-d`, `--device` | enum | No | `cpu` | Export device: `cpu` or `cuda`. Drives ONNX Runtime provider (CPU vs CUDA). |
| **dtype** | `-p`, `--dtype` | enum | No | `fp32` | Export precision: `fp32` or `fp16`. No int8 in this script. |
| **keep_weights** | `--no-keep-weights` (inverted) | boolean | No | true (keep) | If **false** (i.e. `--no-keep-weights`): after a successful export, the script deletes the **local weights directory** used for export. Important for backend to avoid filling disk when using MinIO tar. |
| **use_onnxslim** | `--onnxslim` | boolean | No | false | When true, runs OnnxSlim on each `*.onnx` in the output dir (optimize, then overwrite). Some environments can hit InvalidProtobuf on load after slimming. |

### Backend mapping from “remote model tar from MinIO”

- **Weights source:** Backend should: 1) pull the model tar from MinIO (e.g. `s3://bucket/qwen/llm/qwen-2.5-0.5B-LLM.tar`), 2) extract to a temp or staged dir, 3) pass that **directory path** as the effective “model_id” (or add a dedicated `weights_path` parameter and use it instead of resolving from `model_id`).
- **cache_root:** Script uses `./model_weights` and `model_id` to derive the weights path. For backend, either set `cache_root` to the extracted path and pass a synthetic `model_id`, or refactor to accept `weights_path` explicitly.

---

## Output format

- **Location:** The directory given by `-o` / `--output`.
- **Layout:** Same as a standard Optimum ONNX + Hugging Face tokenizer layout:

```
<output_dir>/
├── config.json              # Model config (from transformers)
├── generation_config.json    # Generation config
├── model.onnx               # Main ONNX graph (name may vary if multi-file)
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt               # If applicable
└── special_tokens_map.json  # If applicable
```

- **Optional:** If `--onnxslim` was used, `*.onnx` files in this dir are overwritten in place (same paths, optimized content).
- **Semantics:** The directory is **self-contained** and loadable with:

  - `ORTModelForCausalLM.from_pretrained(output_dir, export=False, provider=...)`
  - `AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True, ...)`

- **Exit behavior:** Script exits with 0 on success; on exception it raises (no structured JSON output). Backend may wrap this and return success/failure plus paths or error messages.

---

## Flags important for the backend

1. **`-o` / `--output`** — Where to write the ONNX artifact. Backend should set this to a known path (e.g. per-job dir or object key prefix) so it can then upload the result back to MinIO or expose it via API.
2. **`--no-keep-weights`** — **Critical for server:** after a successful export, the script deletes the (downloaded/extracted) weights dir. Use this when the backend extracts the MinIO tar to a temp dir so that dir can be removed after export and disk usage stays bounded.
3. **`-d` / `--device`** — Must match the server’s runtime (e.g. `cuda` if the export runs on a GPU node). Affects export correctness and speed.
4. **`-p` / `--dtype`** — Choose `fp16` for smaller/faster inference when the deployment supports it; `fp32` for maximum compatibility.
5. **`--onnxslim`** — Optional optimization; enable only if your deployment has been tested with OnnxSlim output (some environments report InvalidProtobuf). Backend can make this a configurable option.

---

## Example CLI (current script)

```bash
# Default: Qwen2.5-0.5B, fp32, CPU, keep weights, no OnnxSlim
python qwen-export.py

# Custom model, output dir, fp16, CUDA, discard weights after export, with OnnxSlim
python qwen-export.py Qwen/Qwen2.5-1.5B-Instruct -o ./out_1.5b -d cuda -p fp16 --no-keep-weights --onnxslim
```

---

## Example backend API (conceptual)

For the backend that uses “remote model tar from MinIO”:

- **Inputs to expose:**  
  `minio_tar_uri` (e.g. `s3://bucket/qwen/llm/qwen-2.5-0.5B-LLM.tar`), `output_dir`, `device`, `dtype`, `keep_weights` (or `cleanup_weights_after_export`), `use_onnxslim`.
- **Flow:** Resolve MinIO → download tar → extract to temp dir → call export with `weights_path=extracted_dir` and `output_dir` → optionally upload `output_dir` back to MinIO → if `keep_weights=false`, remove extracted dir.
- **Output:** Success + path or URI of the exported ONNX directory (or list of files); failure + error message.

This spec (input fields, output format, and important flags) is the prototype contract for that backend method.

---

## Model-as-a-Service via Docker

You can run inference as an HTTP service using Docker. **Full deployment guide (conda env → build → run):** [docker/README.md](docker/README.md).

- **Original Qwen (PyTorch):** [docker/original-qwen-service](docker/original-qwen-service) — serves the original model (e.g. from Hugging Face / ModelScope or a mounted path).
- **ONNX Qwen:** [docker/onnx-qwen-service](docker/onnx-qwen-service) — serves an ONNX-exported model (e.g. from `qwen-export.py`).

**Build both images (CPU or CUDA):** `./scripts/build_images.sh cpu` or `./scripts/build_images.sh cuda`. Then run and call `GET /health` and `POST /infer` (optional body `{"prompt":"..."}`). See [docker/README.md](docker/README.md) for step-by-step deployment.
