#!/bin/bash
set -euo pipefail

# Optional hotfix for environments that hit flash-attn rotary dtype assertions
# in Qwen2.5-VL attention (q.float vs cos/sin bf16 mismatch).

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
import pathlib
import shutil
import site
import sys

needle_a = "apply_rotary_emb(q.float(), cos, sin)"
needle_b = "apply_rotary_emb(k.float(), cos, sin)"
patch_a = "apply_rotary_emb(q.float(), cos.float(), sin.float())"
patch_b = "apply_rotary_emb(k.float(), cos.float(), sin.float())"

candidate_paths = []
for base in site.getsitepackages() + [site.getusersitepackages()]:
    candidate_paths.append(
        pathlib.Path(base) / "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
    )

target = None
for path in candidate_paths:
    if path.exists():
        target = path
        break

if target is None:
    print("[hotfix] target file not found in site-packages", file=sys.stderr)
    sys.exit(1)

text = target.read_text(encoding="utf-8")
if patch_a in text and patch_b in text:
    print(f"[hotfix] already patched: {target}")
    sys.exit(0)

if needle_a not in text or needle_b not in text:
    print(f"[hotfix] expected patterns not found: {target}", file=sys.stderr)
    sys.exit(2)

backup = target.with_suffix(target.suffix + ".bak")
if not backup.exists():
    shutil.copy2(target, backup)

text = text.replace(needle_a, patch_a).replace(needle_b, patch_b)
target.write_text(text, encoding="utf-8")
print(f"[hotfix] patched: {target}")
print(f"[hotfix] backup:  {backup}")
PY
