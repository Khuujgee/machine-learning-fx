#!/usr/bin/env bash
# One-time setup: virtualenv + dependencies + .env.
#   ./setup.sh          core install (enough for the scanner with --no-news)
#   ./setup.sh --nlp    also install transformers + torch for FinBERT news sentiment (~2 GB)
set -euo pipefail
cd "$(dirname "$0")"

# Pick a Python the scientific stack ships wheels for (3.9-3.13). Brand-new / pre-release versions
# (e.g. 3.15) make pip try to compile pyarrow, numpy etc. from source, which fails.
pick_python() {
  for c in ${PYTHON:-} python3.13 python3.12 python3.11 python3.10 python3 /usr/bin/python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) and sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
      echo "$c"; return
    fi
  done
  return 1
}
PY="$(pick_python)" || { echo "No supported Python (3.9-3.13) found. Install 3.12 from python.org and re-run."; exit 1; }
rm -rf .venv
echo "==> creating .venv with $("$PY" --version) ($PY)"
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip

echo "==> installing core dependencies"
grep -vE '^(transformers|torch)' requirements.txt > .requirements-core.txt
.venv/bin/pip install -q -r .requirements-core.txt
rm -f .requirements-core.txt

if [[ "${1:-}" == "--nlp" ]]; then
  echo "==> installing FinBERT dependencies (transformers + torch)"
  .venv/bin/pip install -q "transformers>=4.40" "torch>=2.2"
fi

# XGBoost on macOS needs the OpenMP runtime (libomp).
if ! .venv/bin/python -c "import xgboost" 2>/dev/null; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> installing libomp with Homebrew"
    brew install libomp
  else
    echo "==> Homebrew not found: reusing the OpenMP library bundled with scikit-learn"
    SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp "$SITE/sklearn/.dylibs/libomp.dylib" "$SITE/xgboost/lib/"
    install_name_tool -add_rpath @loader_path "$SITE/xgboost/lib/libxgboost.dylib" 2>/dev/null || true
    codesign -f -s - "$SITE/xgboost/lib/libxgboost.dylib" 2>/dev/null
  fi
fi
.venv/bin/python -c "import xgboost, pandas, fastapi; print('==> dependencies OK (xgboost', xgboost.__version__ + ')')"

[[ -f .env ]] || { cp .env.example .env; echo "==> created .env from .env.example"; }

cat <<'EOF'

Setup complete. Next:
  source .venv/bin/activate
  python -m src.data_collection            # ~1 min: download prices, build features
  python -m src.train                      # ~1-2 min: train + walk-forward report
  python -m src.scanner --once --no-news   # one live scan
EOF
