#!/usr/bin/env bash
# Fine-tune several pre-trained CNNs on the same data and print their test accuracy.
# Usage: bash scripts/compare_backbones.sh data/rad
set -euo pipefail
DATA=${1:?"image-folder root required"}

for B in inception_v3 resnet50 vgg16 mobilenet_v2 densenet121; do
  echo ">> $B"
  python -m roadanomaly.train --data "$DATA" --backbone "$B" --out "runs/$B"
done

python - <<'EOF'
import json, pathlib
for run in sorted(pathlib.Path("runs").glob("*/test_metrics.json")):
    print(f"{run.parent.name:<14} test accuracy {100 * json.loads(run.read_text())['accuracy']:.2f}%")
EOF
