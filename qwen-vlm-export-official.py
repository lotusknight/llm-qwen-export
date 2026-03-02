"""使用官方推荐方式（分阶段 torch.onnx.export）导出 Qwen2.5-VL 为 ONNX。

流程：导出 embed_tokens → 解码器（含 past_key_values 的 DynamicCache 封装）→ 视觉编码器，
      再做 shape inference 与可选 onnxslim，最后保存到输出目录。

默认模型：Qwen/Qwen2.5-VL-3B-Instruct。
"""

import argparse
import os
import shutil
from pathlib import Path


def get_model_weights(
    model_id: str, cache_root: str = "./model_weights"
) -> tuple[str, bool]:
    """获取模型权重：有本地缓存则直接用，否则用 ModelScope 下载。返回 (路径, 是否原本就存在)。"""
    safe_name = model_id.replace("/", "--")
    local_path = Path(cache_root) / safe_name

    if local_path.exists() and (local_path / "config.json").exists():
        print(f"✅ 发现本地缓存: {local_path.absolute()}")
        return str(local_path), True

    print("🚀 本地未发现模型，使用 ModelScope 下载...")
    local_path.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download as ms_snapshot

    path = ms_snapshot(model_id=model_id, local_dir=str(local_path))
    return path, False


def _get_model_and_processor(model_id: str, weights_path: str):
    """加载 PyTorch 模型与 processor（Qwen2.5-VL）。"""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(weights_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        weights_path,
        trust_remote_code=True,
    )
    return model, processor


def _make_patched_forward(original_forward, num_hidden_layers: int):
    """构造将 past_key_values 转为 DynamicCache 的 forward，用于 ONNX 导出。"""

    def patched_forward(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        *past_key_values_args,
    ):
        from transformers import DynamicCache

        args_list = list(past_key_values_args)
        if len(args_list) == 0:
            past_key_values = None
        else:
            past_key_values = DynamicCache()
            for i in range(num_hidden_layers):
                key = args_list.pop(0)
                value = args_list.pop(0)
                past_key_values.update(key_states=key, value_states=value, layer_idx=i)

        o = original_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        out = {"logits": o.logits}
        pkv = o.past_key_values
        for i, (key, value) in enumerate(zip(pkv.key_cache, pkv.value_cache)):
            out[f"present.{i}.key"] = key
            out[f"present.{i}.value"] = value
        return out

    return patched_forward


def export_vlm_onnx_official(
    model_id: str,
    output_dir: str,
    *,
    use_onnxslim: bool = False,
    keep_weights: bool = True,
) -> None:
    """
    使用官方推荐的分阶段方式导出 VLM 为 ONNX。
    导出：embed_tokens.onnx、decoder_model_merged.onnx、vision_encoder.onnx。
    """
    import torch

    weights_path, _ = get_model_weights(model_id)
    export_succeeded = False
    temp_dir = Path(output_dir) / "temp"
    final_dir = Path(output_dir) / "onnx"
    temp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, processor = _get_model_and_processor(model_id, weights_path)
        model.eval()

        # 统一取 config：Qwen2.5-VL 有 text_config / vision_config
        config = model.config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            vision_config = config.vision_config
        else:
            text_config = config
            vision_config = getattr(config, "vision_config", None)

        num_heads = text_config.num_attention_heads
        num_key_value_heads = text_config.num_key_value_heads
        num_layers = text_config.num_hidden_layers
        head_dim = text_config.hidden_size // num_heads
        hidden_size = text_config.hidden_size

        if vision_config is None:
            raise ValueError("该模型不是 VLM（无 vision_config），请使用 Qwen2.5-VL。")

        channel = getattr(vision_config, "in_chans", 3)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 1)
        patch_size = getattr(
            vision_config,
            "spatial_patch_size",
            getattr(vision_config, "patch_size", 14),
        )

        # 虚拟输入尺寸
        grid_t, grid_h, grid_w = 1, 16, 16
        batch_size = 1
        sequence_length = 16
        past_sequence_length = 0

        # ----- 1. Embedding 子图 -----
        # Qwen2.5-VL: 文本部分在 model.model.language_model，其有 embed_tokens
        text_model = getattr(model.model, "language_model", model.model)
        embed_module = getattr(
            text_model, "embed_tokens", getattr(text_model, "embed", None)
        )
        if embed_module is None:
            raise AttributeError(
                "无法找到词嵌入模块（language_model.embed_tokens），请确认使用 Qwen2.5-VL 架构。"
            )
        vocab_size = getattr(text_config, "vocab_size", 152064)
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            dtype=torch.int64,
        )
        embed_path = temp_dir / "embed_tokens.onnx"
        torch.onnx.export(
            embed_module,
            (input_ids,),
            str(embed_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            },
            dynamo=False,
        )
        print("✅ 已导出 embed_tokens.onnx")

        # ----- 2. 解码器（带 past_key_values） -----
        # 为当前模型打补丁：forward 接收展平的 past_key_values 并转成 DynamicCache
        num_hidden_layers = num_layers
        original_forward = model.forward
        patched_forward = _make_patched_forward(original_forward, num_hidden_layers)
        model.forward = lambda *a, **k: patched_forward(model, *a, **k)

        inputs_embeds = torch.ones(
            batch_size, sequence_length, hidden_size, dtype=torch.float32
        )
        attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.int64)
        position_ids = torch.ones(3, batch_size, sequence_length, dtype=torch.int64)

        dummy_past = {}
        for i in range(num_layers):
            for key in ("key", "value"):
                dummy_past[f"past_key_values.{i}.{key}"] = torch.zeros(
                    batch_size,
                    num_key_value_heads,
                    past_sequence_length,
                    head_dim,
                    dtype=torch.float32,
                )
        text_inputs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **dummy_past,
        )
        text_inputs_positional = tuple(text_inputs.values())
        decoder_path = temp_dir / "decoder_model_merged.onnx"
        torch.onnx.export(
            model,
            text_inputs_positional,
            str(decoder_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=list(text_inputs.keys()),
            output_names=["logits"]
            + [
                f"present.{i}.{key}"
                for i in range(num_layers)
                for key in ("key", "value")
            ],
            dynamic_axes={
                "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "position_ids": {1: "batch_size", 2: "sequence_length"},
                **{
                    f"past_key_values.{i}.{key}": {
                        0: "batch_size",
                        2: "past_sequence_length",
                    }
                    for i in range(num_layers)
                    for key in ("key", "value")
                },
                "logits": {0: "batch_size", 1: "sequence_length"},
                **{
                    f"present.{i}.{key}": {
                        0: "batch_size",
                        2: "past_sequence_length_plus_1",
                    }
                    for i in range(num_layers)
                    for key in ("key", "value")
                },
            },
            dynamo=False,
        )
        model.forward = original_forward
        print("✅ 已导出 decoder_model_merged.onnx")

        # ----- 3. 视觉编码器 -----
        visual = getattr(model, "visual", getattr(model.model, "visual", None))
        if visual is None:
            raise AttributeError("无法找到视觉编码器（model.visual / model.model.visual）。")
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)
        pixel_values = torch.randn(
            batch_size * grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
            dtype=torch.float32,
        )
        vision_inputs = dict(pixel_values=pixel_values, grid_thw=grid_thw)
        vision_path = temp_dir / "vision_encoder.onnx"
        torch.onnx.export(
            visual,
            tuple(vision_inputs.values()),
            str(vision_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=list(vision_inputs.keys()),
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch_size_x_grid_t_h_w", 1: "channels"},
                "grid_thw": {0: "batch_size"},
                "image_features": {0: "batch_size_x_grid_t_h_w"},
            },
            dynamo=False,
        )
        print("✅ 已导出 vision_encoder.onnx")

        # ----- 4. 保存 config / processor -----
        model.config.save_pretrained(str(Path(output_dir)))
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(str(Path(output_dir)))
        processor.save_pretrained(str(Path(output_dir)))

        # ----- 5. 后处理：shape inference + 可选 onnxslim，写入 final_dir -----
        import onnx
        from optimum.onnx.graph_transformations import check_and_save_model

        for name in (
            "embed_tokens.onnx",
            "decoder_model_merged.onnx",
            "vision_encoder.onnx",
        ):
            temp_path = temp_dir / name
            onnx.shape_inference.infer_shapes_path(
                str(temp_path), check_type=True, strict_mode=True
            )
            try:
                if use_onnxslim:
                    import onnxslim

                    onnx_model = onnxslim.slim(str(temp_path))
                else:
                    onnx_model = onnx.load(str(temp_path))
            except Exception as e:
                print(f"   ⚠️ onnxslim 跳过 {name}: {e}")
                onnx_model = onnx.load(str(temp_path))
            check_and_save_model(onnx_model, str(final_dir / name))

        print("\n✨ 导出完成。ONNX 文件在:", final_dir)
        export_succeeded = True
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
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
        description="使用官方推荐方式（分阶段 torch.onnx.export）导出 Qwen-VL 为 ONNX。"
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="模型 ID，默认 Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./qwen25_vl_3b_onnx_official",
        help="输出目录（其下会创建 onnx/ 存放 ONNX 文件）",
    )
    parser.add_argument(
        "--onnxslim",
        action="store_true",
        help="对 ONNX 做 onnxslim 优化",
    )
    parser.add_argument(
        "--no-keep-weights",
        action="store_true",
        help="导出成功后删除下载的权重目录",
    )
    args = parser.parse_args()

    export_vlm_onnx_official(
        args.model_id,
        args.output,
        use_onnxslim=args.onnxslim,
        keep_weights=not args.no_keep_weights,
    )


if __name__ == "__main__":
    main()
