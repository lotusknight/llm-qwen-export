# ONNX Qwen

Inference over ONNX-exported Qwen. Requires ONNX dir mounted at `/model`. Build/deploy: see [../COMMANDS.md](../COMMANDS.md).

| Env               | Default | Description |
|-------------------|--------|-------------|
| `ONNX_MODEL_PATH` | /model | Path in container to ONNX export dir |
| `DEVICE`          | cpu    | cpu \| cuda |
| `PORT`            | 8000   | Listen port |
| `MAX_NEW_TOKENS`  | 128    | Max new tokens |
| `PRELOAD_MODEL`   | 1      | 0 = skip load at startup |

**Endpoints:** `GET /health`, `POST /infer` (body optional: `{"prompt":"..."}`).
