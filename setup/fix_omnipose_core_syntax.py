#!/usr/bin/env python3
"""
Fix omnipose/core.py SyntaxError (unterminated string at no-cell-pixels diagnostic).
Run from build script with path to the core.py to fix (e.g. app bundle's copy).
Usage: python fix_omnipose_core_syntax.py <path_to_omnipose_core_py>
"""
import re
import sys
from pathlib import Path


def main() -> bool:
    if len(sys.argv) < 2:
        print("Usage: fix_omnipose_core_syntax.py <path_to_omnipose/core.py>", file=sys.stderr)
        return False
    core_path = Path(sys.argv[1]).resolve()
    if not core_path.is_file():
        print(f"Not a file: {core_path}", file=sys.stderr)
        return False

    content = core_path.read_text(encoding="utf-8")
    if "\r" in content:
        content = content.replace("\r\n", "\n").replace("\r", "\n")

    one_line = '                    _f.write("[omnipose no cell pixels] dist min={:.4f} max={:.4f} mean={:.4f} shape={} mask_thr={}\\n".format(float(np.min(_d)), float(np.max(_d)), float(np.mean(_d)), _d.shape, mask_threshold))'
    if one_line in content:
        print(f"  omnipose/core.py already fixed: {core_path}")
        return True

    if "omnipose no cell pixels" not in content:
        print(f"  omnipose/core.py has no diagnostic block: {core_path}")
        return True

    # Match any variant: optional leading whitespace (e.g. app bundle has many spaces), then _f.write... until mask_threshold))
    pattern = re.compile(
        r'\s*_f\.write\s*\(\s*"\[omnipose no cell pixels\] dist min=\{\:.4f\} max=\{\:.4f\} mean=\{\:.4f\} shape=\{\} mask_thr=\{\}[\s\S]*?mask_threshold\s*\)\s*\)',
        re.DOTALL,
    )
    if not pattern.search(content):
        print(f"  Could not find broken pattern in {core_path}", file=sys.stderr)
        return False

    new_content = pattern.sub(one_line, content, count=1)
    core_path.write_text(new_content, encoding="utf-8")
    print(f"  Fixed omnipose/core.py (one-line diagnostic): {core_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
