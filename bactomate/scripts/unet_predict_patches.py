#!/usr/bin/env python
"""
Run U-Net patch-based prediction on a folder. Uses in-repo TensorFlow/Keras implementation.
Requires: pip install tensorflow
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    parser = argparse.ArgumentParser(description="Segment images with patch-based U-Net.")
    parser.add_argument("-i", "--input_folder", type=Path, required=True, help="Folder with images")
    parser.add_argument("-o", "--output_folder", type=Path, required=True, help="Output folder")
    parser.add_argument("-m", "--model_path", type=Path, required=True, help="Path to .keras model")
    parser.add_argument("--config-path", type=Path, default=None, help="patch_config.json (default: next to model)")
    args = parser.parse_args()

    config_path = args.config_path or (
        args.model_path.parent / f"{args.model_path.stem}_patch_config.json"
    )
    if not args.model_path.is_file():
        print(f"Error: Model not found: {args.model_path}", file=sys.stderr)
        return 1
    if not args.input_folder.is_dir():
        print(f"Error: Input folder not found: {args.input_folder}", file=sys.stderr)
        return 1

    try:
        from bactomate.scripts import unet_predict_impl
        return unet_predict_impl.run(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            model_path=args.model_path,
            config_path=config_path,
        )
    except ImportError as e:
        python_exe = getattr(sys, "executable", "python")
        print(
            "U-Net prediction requires TensorFlow. Install it for the same Python that runs BactoMate:",
            file=sys.stderr,
        )
        print(f"  {python_exe} -m pip install tensorflow", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
