#!/usr/bin/env bash
# Example AgentDojo benchmark run for the Full-CAD classifier defense.
# Requires outputs/models/classifier_fullcad.pt and an API key in .env or the environment.
# Run from the artifact root:
#   bash scripts/run_agentdojo_benchmark.sh

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

export PYTHONPATH="$REPO_ROOT/third_party/AutoDojo/agentdojo/src:$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export AGENTDOJO_DEFENSE_PLUGINS=detector
export EMBED_MODEL="${EMBED_MODEL:-jinaai/jina-embeddings-v3}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda:0}"
export CLASSIFIER_DEVICE="${CLASSIFIER_DEVICE:-cuda:0}"
export CLASSIFIER_WEIGHT_PATH="${CLASSIFIER_WEIGHT_PATH:-$REPO_ROOT/outputs/models/classifier_fullcad.pt}"
export CLASSIFIER_THRESHOLD="${CLASSIFIER_THRESHOLD:-0.5}"

[ -f "$CLASSIFIER_WEIGHT_PATH" ] || {
  echo "Missing classifier checkpoint: $CLASSIFIER_WEIGHT_PATH" >&2
  echo "Run: bash scripts/rebuild_classifier.sh" >&2
  exit 1
}

MODEL="${MODEL:-openai/gpt-4o-mini}"
LOGDIR="${LOGDIR:-$REPO_ROOT/runs/full_cad}"
SUITES=("${@:-banking}")

mkdir -p "$LOGDIR"

for SUITE in "${SUITES[@]}"; do
  python -m agentdojo.scripts.benchmark \
    --model "$MODEL" \
    --suite "$SUITE" \
    --defense full_cad \
    --logdir "$LOGDIR"

  python -m agentdojo.scripts.benchmark \
    --model "$MODEL" \
    --suite "$SUITE" \
    --attack important_instructions \
    --defense full_cad \
    --logdir "$LOGDIR"

  CACHE="$REPO_ROOT/third_party/AutoDojo/agentdojo/variant_generation/variants/${SUITE}/${MODEL}/cls/injections.json"
  if [ -f "$CACHE" ]; then
    AUTODOJO_CACHE="$CACHE" AUTODOJO_VARIANT="${AUTODOJO_VARIANT:-0}" \
      python -m agentdojo.scripts.benchmark \
        --model "$MODEL" \
        --suite "$SUITE" \
        --attack autodojo \
        --defense full_cad \
        --logdir "$LOGDIR"
  else
    echo "Skipping AutoDojo transfer attack; cache not found: $CACHE" >&2
  fi
done

echo "Wrote benchmark logs under $LOGDIR"
