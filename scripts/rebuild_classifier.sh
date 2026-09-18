#!/usr/bin/env bash
# Rebuild the Full-CAD embeddings and classifier from the JSON data in data/
# (downloaded from Hugging Face; see README).
# Run from the artifact root:
#   bash scripts/rebuild_classifier.sh

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cache_dir="${CACHE_ROOT:-$REPO_ROOT/.cache}"
export HF_HOME="$cache_dir"
export TORCH_HOME="$cache_dir"
export TRITON_CACHE_DIR="$cache_dir"
export TORCH_EXTENSIONS_DIR="$cache_dir"
export MPLCONFIGDIR="$cache_dir/matplotlib"
export TMPDIR="$cache_dir"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT/third_party/AutoDojo/agentdojo/src:$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$cache_dir" "$MPLCONFIGDIR" outputs/embeddings outputs/models

python -m detector.build_embeddings \
  --input_dir data/training_data_original \
  --output_pt outputs/embeddings/original_fullcad.pt

python -m detector.build_embeddings \
  --input_dir data/testing_data_original \
  --output_pt outputs/embeddings/test_fullcad.pt

python -m detector.train_and_eval \
  outputs/embeddings/original_fullcad.pt outputs/embeddings/test_fullcad.pt \
  --test_dir data/testing_data_original \
  --test_domains banking travel slack \
  --save_path outputs/models/classifier_fullcad.pt \
  --seed "${SEED:-0}"

echo "Wrote outputs/models/classifier_fullcad.pt"

# Paraphrase-bank embeddings consumed by adversarial_training.llm_paraphrase
# (--para-files). Built per suite from the paraphrased JSON.
for suite in banking slack travel; do
  python -m detector.build_embeddings \
    --input_dir "data/training_data_paraphrased/agentdojo_${suite}_scenarios" \
    --output_pt "outputs/embeddings/para_${suite}_fullcad.pt"
done

echo "Wrote outputs/embeddings/para_{banking,slack,travel}_fullcad.pt"
