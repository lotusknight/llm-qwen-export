# Original Qwen (PyTorch)

Inference over PyTorch/Transformers Qwen. Build/deploy: see [../COMMANDS.md](../COMMANDS.md).

| Env           | Default              | Description |
|---------------|----------------------|-------------|
| `MODEL_ID`    | Qwen/Qwen2.5-0.5B-Instruct | Model to download if not using `MODEL_PATH` |
| `MODEL_PATH`  | (empty)              | Path in container to local weights dir (overrides `MODEL_ID`) |
| `HF_ENDPOINT` | https://hf-mirror.com | Hugging Face endpoint; empty = default |
| `DEVICE`      | cpu                  | cpu \| cuda |
| `PORT`        | 8000                 | Listen port |
| `MAX_NEW_TOKENS` | 128              | Max new tokens |
| `PRELOAD_MODEL`  | 0                 | 1 = load at startup |

**Endpoints:** `GET /health`, `POST /infer` (body optional: `{"prompt":"..."}`).
