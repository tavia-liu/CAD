#!/usr/bin/env bash
# Generate the paraphrase bank via OpenRouter. Reads OPENROUTER_API_KEY from
# the repo-root .env. Paths are resolved by the Python script, so run from
# anywhere.
#   bash src/adversarial_training/paraphrase_generation/run.sh
#   PARA_MODEL=anthropic/claude-sonnet-4-6 bash src/adversarial_training/paraphrase_generation/run.sh
#   bash src/adversarial_training/paraphrase_generation/run.sh --limit 4 --dry-run
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$(dirname "$(dirname "$HERE")")")"
set -a; source "$REPO/.env"; set +a
export PARA_PROVIDER="${PARA_PROVIDER:-openrouter}"
export PARA_MODEL="${PARA_MODEL:-deepseek/deepseek-v4-flash}"
PYTHONPATH="$REPO/third_party/AutoDojo/agentdojo/src:$REPO/src:$REPO:${PYTHONPATH:-}" \
python -m adversarial_training.paraphrase_generation.gen_paraphrase_bank --workers "${WORKERS:-8}" "$@"
