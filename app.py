import argparse
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from cutout_tool import build_transform, infer_mask, load_model, merge_protection, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="BiRefNet cutout web UI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="birefnet_finetuned_toonout.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--size", type=int, default=1024, help="Model input size")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="UI host")
    parser.add_argument("--port", type=int, default=7860, help="UI port")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = resolve_device(args.device)
    model = load_model(checkpoint, device)
    transform = build_transform(args.size)

    def create_editor_background(image: Image.Image | None):
        if image is None:
            return {"background": None, "layers": [], "composite": None}
        rgba = image.convert("RGBA")
        return {"background": rgba, "layers": [], "composite": rgba}

    def extract_mask_from_editor(editor_value: Any, target_size: tuple[int, int]) -> np.ndarray | None:
        if editor_value is None:
            return None

        if isinstance(editor_value, Image.Image):
            return np.array(editor_value.convert("L").resize(target_size, resample=Image.NEAREST), dtype=np.uint8)

        if not isinstance(editor_value, dict):
            return None

        layers = editor_value.get("layers") or []
        merged = None
        for layer in layers:
            if layer is None:
                continue
            layer_rgba = layer.convert("RGBA") if isinstance(layer, Image.Image) else Image.fromarray(layer).convert("RGBA")
            layer_alpha = np.array(layer_rgba, dtype=np.uint8)[..., 3]
            merged = layer_alpha if merged is None else np.maximum(merged, layer_alpha)

        if merged is None:
            return None

        return np.array(Image.fromarray(merged, mode="L").resize(target_size, resample=Image.NEAREST), dtype=np.uint8)

    def predict(image: Image.Image, protect_editor: Any, threshold: int, preserve_input_alpha: bool):
        if image is None:
            return None, None

        image_rgba = image.convert("RGBA")
        image_rgb = image_rgba.convert("RGB")
        input_alpha = np.array(image_rgba, dtype=np.uint8)[..., 3]

        mask = infer_mask(model, image_rgb, transform, device)
        user_protect = extract_mask_from_editor(protect_editor, image_rgb.size)

        preserve_alpha = input_alpha if preserve_input_alpha else None
        mask = merge_protection(mask, preserve_alpha_mask=preserve_alpha, user_protect_mask=user_protect)

        if threshold > 0:
            mask = np.where(mask >= threshold, mask, 0).astype(np.uint8)

        rgba = np.array(image_rgb.convert("RGBA"))
        rgba[..., 3] = mask

        return Image.fromarray(rgba, mode="RGBA"), Image.fromarray(mask, mode="L")

    with gr.Blocks(title="BiRefNet Cutout") as demo:
        gr.Markdown("# BiRefNet 切り抜き\n画像を入れると透過PNGを作成します。")
        with gr.Row():
            input_image = gr.Image(type="pil", label="入力画像")
            protect_editor = gr.ImageEditor(
                type="pil",
                image_mode="RGBA",
                label="保護マスクを描く(任意/白=保護)",
                brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed"),
                eraser=gr.Eraser(),
                layers=True,
                transforms=(),
            )
            output_image = gr.Image(type="pil", label="透過結果 (RGBA)")
        mask_image = gr.Image(type="pil", label="マスク (Gray)")
        threshold = gr.Slider(minimum=0, maximum=255, step=1, value=0, label="Alpha Threshold")
        preserve_input_alpha = gr.Checkbox(value=True, label="入力画像の既存アルファを保護")
        run_button = gr.Button("切り抜き実行")

        input_image.change(fn=create_editor_background, inputs=[input_image], outputs=[protect_editor])

        run_button.click(
            fn=predict,
            inputs=[input_image, protect_editor, threshold, preserve_input_alpha],
            outputs=[output_image, mask_image],
        )

    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()