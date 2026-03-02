# Build, deploy, test — both services

Run from **project root** unless noted. Use `sudo` for docker if needed.

---

## 1. Build

```bash
sudo ./scripts/build_images.sh cpu
# or
sudo ./scripts/build_images.sh cuda
```

Images: `qwen-original-inference:cpu|cuda`, `qwen-onnx-inference:cpu|cuda`.

**Single image:**

```bash
sudo docker build --build-arg RUNTIME=cuda -t qwen-original-inference:cuda \
  -f docker/original-qwen-service/Dockerfile docker/original-qwen-service

sudo docker build --build-arg RUNTIME=cuda -t qwen-onnx-inference:cuda \
  -f docker/onnx-qwen-service/Dockerfile docker/onnx-qwen-service
```

---

## 2. Deploy

### Original Qwen (PyTorch)

```bash
# Download at first /infer (HF mirror in image)
sudo docker run -p 8038:8000 -e MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct qwen-original-inference:cuda

# Local model (no download)
sudo docker run -p 8038:8000 -v /path/to/weights:/model -e MODEL_PATH=/model qwen-original-inference:cuda
```

GPU: add `--gpus all -e DEVICE=cuda`.

### ONNX Qwen

Prereq: ONNX export dir (e.g. from `qwen-export.py -o <dir>`). Mount at `/model`.

```bash
sudo docker run -p 8038:8000 -v /path/to/onnx_export:/model qwen-onnx-inference:cuda
sudo docker run --gpus all -p 8038:8000 -v /path/to/onnx_export:/model -e DEVICE=cuda qwen-onnx-inference:cuda
```

---

## 3. Test

```bash
curl http://localhost:8038/health
curl -X POST http://localhost:8038/infer -H "Content-Type: application/json" -d '{"prompt":"Hello"}'
```

Response: `{"reply":"...", "prompt_used":"..."}`.

---

## 4. Containers

```bash
sudo docker ps -a
sudo docker rm -f <id_or_name>
sudo docker container prune
```
