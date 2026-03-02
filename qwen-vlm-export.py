"""将 Qwen2.5-VL（视觉语言模型）导出为 ONNX。

与纯文本 Qwen 不同：VLM 是 image-to-text，输入含图像 + 文本，使用 ORTModelForVision2Seq。
Optimum 目前不原生支持 qwen2_vl / qwen2_5_vl，本地 export 会报错；需用其他工具导出。

其他导出方式（不拉取预导出 ONNX，用工具自行导出）：
1. 官方推荐：ModelScope / 魔搭社区
   - 阿里官方在 ModelScope 对 Qwen-VL 系列有适配（2D-RoPE、多动态分辨率）。
   - 可查 ModelScope 文档、Qwen 官方仓库与 PAI-DSW 等是否有导出脚本或 Notebook。
   - ModelScope: https://modelscope.cn  |  Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
2. Optimum 自定义导出：为 qwen2_5_vl 实现 custom_onnx_configs 后传入 from_pretrained。
   - 文档: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models
3. 分阶段导出：用 torch.onnx.export 分别导出视觉编码器 + 语言模型，再拼成完整流程。
4. LMDeploy 等：Qwen2.5-VL 支持 LMDeploy 部署（非 ONNX，可作替代方案）。
默认尝试模型：Qwen/Qwen2.5-VL-3B-Instruct。
"""

import argparse
import sys
import shutil
from pathlib import Path

from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

# Optimum 报错中的关键词，用于识别「不支持该架构」
_UNSUPPORTED_ARCH_MSGS = (
    "custom or unsupported architecture",
    "custom_onnx_configs",
    "qwen2_vl",
    "qwen2_5_vl",
)


def get_model_weights(
    model_id: str, cache_root: str = "./model_weights"
) -> tuple[str, bool]:
    """获取模型权重：有本地缓存则直接用，否则用 ModelScope 下载。返回 (路径, 是否原本就存在)。"""
    safe_name = model_id.replace("/", "--")
    local_path = Path(cache_root) / safe_name

    # 已有完整缓存则直接返回
    if local_path.exists() and (local_path / "config.json").exists():
        print(f"✅ 发现本地缓存: {local_path.absolute()}")
        return str(local_path), True

    print("🚀 本地未发现模型，使用 ModelScope 下载...")
    local_path.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download as ms_snapshot

    path = ms_snapshot(model_id=model_id, local_dir=str(local_path))
    return path, False


def _provider_for_device(device: str) -> str:
    """根据 device 字符串返回 ONNX Runtime 的 provider。"""
    if device.lower() == "cuda":
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def export_to_onnx(
    model_id: str,
    output_dir: str,
    *,
    device: str = "cpu",
    dtype: str = "fp32",
    keep_weights: bool = True,
) -> None:
    """
    导出主流程：拉取/加载权重 → 导出 VLM ONNX；若不支持则打印其他导出方式并退出。
    device: cpu / cuda；dtype: fp32 / fp16；keep_weights 为 False 且导出成功时删除权重目录。
    """
    weights_path, _ = get_model_weights(model_id)
    export_succeeded = False
    provider = _provider_for_device(device)

    try:
        # 1. 导出 ONNX（VLM：image-to-text，含视觉编码器 + 语言模型）
        print(f"\n📦 导出 VLM ONNX 至: {output_dir} (device={device}, dtype={dtype})")
        model = ORTModelForVision2Seq.from_pretrained(
            weights_path,
            export=True,
            trust_remote_code=True,
            provider=provider,
            dtype=dtype if dtype in ("fp32", "fp16", "bf16") else "fp32",
        )
        model.save_pretrained(output_dir)
        processor = AutoProcessor.from_pretrained(weights_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)

        print("\n✨ 导出完成。")
        export_succeeded = True
    except ValueError as e:
        # Optimum 不支持 qwen2_vl / qwen2_5_vl 时给出替代方案说明
        err = str(e).lower()
        if any(msg in err for msg in _UNSUPPORTED_ARCH_MSGS):
            print(
                "\n⚠️ Optimum 暂不支持 qwen2_vl / qwen2_5_vl 原生 ONNX 导出。",
                file=sys.stderr,
            )
            print(
                "  请改用其他方式导出（见本文件顶部文档）：",
                file=sys.stderr,
            )
            print(
                "  1) 官方推荐：ModelScope/魔搭 导出工具（2D-RoPE、多动态分辨率适配）",
                file=sys.stderr,
            )
            print(
                "  2) Optimum 自定义：实现 custom_onnx_configs 传入 from_pretrained",
                file=sys.stderr,
            )
            print(
                "  3) 分阶段：torch.onnx.export 分别导出视觉编码器与语言模型",
                file=sys.stderr,
            )
            print(
                "  4) LMDeploy 等非 ONNX 部署方案",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        raise
    finally:
        # 2. 可选清理：仅当导出成功且 keep_weights=False 时删除权重目录
        if not keep_weights and export_succeeded:
            print(f"\n🗑️ 清理权重: {weights_path}")
            try:
                shutil.rmtree(weights_path)
                print("✅ 已清理。")
            except OSError as e:
                print(f"❌ 清理失败: {e}")
        elif keep_weights:
            print(f"\n💾 权重保留: {weights_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 Qwen2.5-VL 视觉语言模型导出为 ONNX（image-to-text）。"
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="模型 ID，默认 Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./qwen25_vl_3b_onnx",
        help="ONNX 输出目录，默认 ./qwen25_vl_3b_onnx",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="设备：cpu 或 cuda，默认 cpu",
    )
    parser.add_argument(
        "-p",
        "--dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="精度：fp32（全精度）或 fp16（半精度），默认 fp32",
    )
    parser.add_argument(
        "--no-keep-weights",
        action="store_true",
        help="导出成功后删除下载的权重目录以节省空间",
    )
    args = parser.parse_args()

    export_to_onnx(
        args.model_id,
        args.output,
        device=args.device,
        dtype=args.dtype,
        keep_weights=not args.no_keep_weights,
    )


if __name__ == "__main__":
    main()
