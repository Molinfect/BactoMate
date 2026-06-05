#!/usr/bin/env python
"""
Run U-Net patch-based training. Uses in-repo TensorFlow/Keras implementation.
Requires: pip install tensorflow
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    # Reduce TensorFlow/Metal log spam (INFO, WARNING, and some optimizer errors) before TF loads
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=errors only
    parser = argparse.ArgumentParser(description="Train U-Net on patches.")
    parser.add_argument("--train-path", type=Path, required=True, help="Training directory")
    parser.add_argument("--config-path", type=Path, required=True, help="patch_config.json path")
    parser.add_argument("--output-model-path", type=Path, required=True, help="Output .keras model path")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--use-gpu", type=int, default=1, help="1 = use GPU if available")
    parser.add_argument("--img-filter", default="", help="Image filename suffix")
    parser.add_argument("--mask-filter", default="_mask", help="Mask filename suffix")
    args = parser.parse_args()

    try:
        from bactomate.scripts import unet_train_impl
        return unet_train_impl.run(
            train_path=args.train_path,
            config_path=args.config_path,
            output_model_path=args.output_model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_gpu=bool(args.use_gpu),
            img_filter=args.img_filter,
            mask_filter=args.mask_filter,
        )
    except ImportError as e:
        python_exe = getattr(sys, "executable", "python")
        print(
            "U-Net training requires TensorFlow. Install it for the same Python that runs BactoMate:",
            file=sys.stderr,
        )
        print(f"  {python_exe} -m pip install tensorflow", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        return 1
    except (OSError, Exception) as e:
        err_msg = str(e).lower()
        if "tensorflow" in err_msg and ("metal" in err_msg or "dlopen" in err_msg or "load" in err_msg):
            python_exe = getattr(sys, "executable", "python")
            print(
                "TensorFlow failed to load (often due to tensorflow-metal version mismatch).",
                file=sys.stderr,
            )
            print(
                "Option A — train on CPU: uninstall the Metal plugin, then try again:",
                file=sys.stderr,
            )
            print(f"  {python_exe} -m pip uninstall tensorflow-metal -y", file=sys.stderr)
            print(
                "Option B — use GPU: reinstall matching TensorFlow + tensorflow-metal (e.g. both 2.15):",
                file=sys.stderr,
            )
            print(f"  {python_exe} -m pip install tensorflow==2.15.0 tensorflow-metal==1.1.0", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
