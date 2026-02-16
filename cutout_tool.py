import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from birefnet.models.birefnet import BiRefNet


def check_state_dict(state_dict, unwanted_prefixes=("module.", "_orig_mod.")):
    for key, value in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if key[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[key[prefix_length:]] = state_dict.pop(key)
    return state_dict


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def collect_images(input_path: Path) -> list[Path]:
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if input_path.is_file():
        if input_path.suffix.lower() not in valid_exts:
            raise ValueError(f"Unsupported input file extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in valid_exts]
    if not files:
        raise FileNotFoundError(f"No images found under: {input_path}")
    return files


def load_model(checkpoint: Path, device: torch.device) -> BiRefNet:
    model = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()
    return model


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def infer_mask(
    model: BiRefNet,
    pil_image: Image.Image,
    transform,
    device: torch.device,
) -> np.ndarray:
    width, height = pil_image.size
    tensor = transform(pil_image).unsqueeze(0).to(device)
    if device.type == "cuda":
        tensor = tensor.half()

    with torch.no_grad():
        pred = model(tensor)[-1].sigmoid()
        pred = torch.nn.functional.interpolate(
            pred,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )

    mask = (pred.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return mask


def resolve_protect_path(source_image: Path, input_root: Path, protect_path: Path | None) -> Path | None:
    if protect_path is None:
        return None

    if protect_path.is_file():
        return protect_path

    if protect_path.is_dir() and input_root.is_dir():
        rel = source_image.relative_to(input_root)
        candidate = (protect_path / rel).with_suffix(".png")
        if candidate.exists():
            return candidate
        return None

    return None


def load_protection_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    protect_img = Image.open(path).convert("L")
    protect_img = protect_img.resize(size, resample=Image.NEAREST)
    return np.array(protect_img, dtype=np.uint8)


def merge_protection(
    pred_mask: np.ndarray,
    preserve_alpha_mask: np.ndarray | None = None,
    user_protect_mask: np.ndarray | None = None,
) -> np.ndarray:
    merged = pred_mask
    if preserve_alpha_mask is not None:
        merged = np.maximum(merged, preserve_alpha_mask.astype(np.uint8))
    if user_protect_mask is not None:
        merged = np.maximum(merged, user_protect_mask.astype(np.uint8))
    return merged


def apply_alpha(pil_image: Image.Image, mask: np.ndarray, threshold: int) -> Image.Image:
    rgba = pil_image.convert("RGBA")
    rgba_arr = np.array(rgba)
    if threshold > 0:
        mask = np.where(mask >= threshold, mask, 0).astype(np.uint8)
    rgba_arr[..., 3] = mask
    return Image.fromarray(rgba_arr, mode="RGBA")


def output_path_for(source: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root if output_root.suffix.lower() == ".png" else output_root / f"{source.stem}.png"
    relative = source.relative_to(input_root)
    return (output_root / relative).with_suffix(".png")


def parse_args():
    parser = argparse.ArgumentParser(description="BiRefNet cutout tool: image/folder to transparent PNG")
    parser.add_argument("--input", required=True, type=str, help="Input image path or folder path")
    parser.add_argument("--output", required=True, type=str, help="Output PNG path (single) or folder (batch)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="birefnet_finetuned_toonout.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--size", type=int, default=1024, help="Model input size (default: 1024)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Alpha threshold 0-255 (set pixels below threshold to transparent)",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save grayscale mask next to output PNG as *_mask.png",
    )
    parser.add_argument(
        "--preserve-input-alpha",
        action="store_true",
        default=True,
        help="Preserve opaque regions from input alpha channel as protection (default: enabled)",
    )
    parser.add_argument(
        "--no-preserve-input-alpha",
        action="store_false",
        dest="preserve_input_alpha",
        help="Disable preserving input alpha channel",
    )
    parser.add_argument(
        "--protect",
        type=str,
        default=None,
        help="Protection mask path (single file) or folder (batch). White area is protected.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    protect_path = Path(args.protect).expanduser().resolve() if args.protect else None

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if protect_path and not protect_path.exists():
        raise FileNotFoundError(f"Protection path not found: {protect_path}")
    if args.threshold < 0 or args.threshold > 255:
        raise ValueError("--threshold must be in 0..255")

    images = collect_images(input_path)
    device = resolve_device(args.device)
    print(f"Running on device: {device}")
    model = load_model(checkpoint, device)
    transform = build_transform(args.size)

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    elif output_path.suffix.lower() != ".png":
        output_path.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc="Cutout"):
        try:
            rgba_input = Image.open(img_path).convert("RGBA")
        except Exception:
            print(f"[skip] Failed to read: {img_path}")
            continue
        pil_image = rgba_input.convert("RGB")
        alpha_input = np.array(rgba_input, dtype=np.uint8)[..., 3]

        mask = infer_mask(model, pil_image, transform, device)
        preserve_alpha = alpha_input if args.preserve_input_alpha else None

        protect_file = resolve_protect_path(img_path, input_path, protect_path)
        user_protect = load_protection_mask(protect_file, pil_image.size) if protect_file else None

        mask = merge_protection(mask, preserve_alpha_mask=preserve_alpha, user_protect_mask=user_protect)
        out_rgba = apply_alpha(pil_image, mask, args.threshold)

        out_file = output_path_for(img_path, input_path, output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_rgba.save(out_file)

        if args.save_mask:
            mask_path = out_file.with_name(f"{out_file.stem}_mask.png")
            Image.fromarray(mask, mode="L").save(mask_path)

    print("Done")


if __name__ == "__main__":
    main()