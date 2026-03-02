#!/usr/bin/env bash
# Build Docker images for original Qwen and ONNX Qwen inference services.
# Usage: ./scripts/build_images.sh [cpu|cuda] [tag_suffix]
# Example: ./scripts/build_images.sh cpu
#          ./scripts/build_images.sh cuda latest-gpu

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME="${1:-cpu}"
TAG_SUFFIX="${2:-}"

if [[ "$RUNTIME" != "cpu" && "$RUNTIME" != "cuda" ]]; then
    echo "Usage: $0 [cpu|cuda] [tag_suffix]"
    exit 1
fi

TAG="$RUNTIME"
[[ -n "$TAG_SUFFIX" ]] && TAG="${TAG}-${TAG_SUFFIX}"

echo "Building images with RUNTIME=$RUNTIME, tag suffix=$TAG_SUFFIX"

docker build \
    --build-arg RUNTIME="$RUNTIME" \
    -t "qwen-original-inference:$TAG" \
    -f "$REPO_ROOT/docker/original-qwen-service/Dockerfile" \
    "$REPO_ROOT/docker/original-qwen-service"

docker build \
    --build-arg RUNTIME="$RUNTIME" \
    -t "qwen-onnx-inference:$TAG" \
    -f "$REPO_ROOT/docker/onnx-qwen-service/Dockerfile" \
    "$REPO_ROOT/docker/onnx-qwen-service"

echo "Done. Images: qwen-original-inference:$TAG, qwen-onnx-inference:$TAG"
