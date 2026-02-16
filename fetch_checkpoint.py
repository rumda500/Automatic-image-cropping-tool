import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download BiRefNet ToonOut checkpoint")
    parser.add_argument(
        "--output",
        type=str,
        default="birefnet_finetuned_toonout.pth",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="joelseytre/toonout",
        help="Hugging Face model repository id",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="birefnet_finetuned_toonout.pth",
        help="Filename in the model repository",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Checkpoint already exists: {output_path}")
        return

    print(f"Downloading checkpoint from {args.repo_id}/{args.filename} ...")
    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(output_path.parent),
    )

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != output_path:
        downloaded_path.replace(output_path)

    print(f"Saved checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
