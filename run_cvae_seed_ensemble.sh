#!/usr/bin/env bash
# Train N seeds of the same CVAE config in parallel-on-CPU / serial-on-GPU.
# Usage: run_cvae_seed_ensemble.sh META_PREFIX SPLIT_DIR N_SEEDS [extra_train_args...]
#
# Example:
#   bash run_cvae_seed_ensemble.sh cvae_Fpca95_seeds \
#        checkpoints/vessel_aware_cvae/splits_only3999_20260427_223850 5 \
#        --model_type v8_resnet --use_conditional_prior \
#        --hidden_dim 160 --latent_dim 24 --encoder_blocks 2 --decoder_blocks 4 \
#        --dropout 0.10 --condition_dropout 0.20 --lr 0.0006 --w_kl 0.004 \
#        --kl_schedule linear --kl_warmup 100 --free_bits 0.05 --pca_var 0.95
set -euo pipefail
PREFIX=$1; SPLIT=$2; N=$3; shift 3
EXTRA="$*"
stamp=$(date +%Y%m%d_%H%M%S)
for s in $(seq 1 "$N"); do
  meta="${PREFIX}_seed${s}_${stamp}"
  log="logs/${meta}.log"
  echo "[seed $s] meta=$meta"
  conda run --no-capture-output -n unified_env python train_vessel_cvae_coeff_trainval.py \
    --train_cases_file "$SPLIT/cases_train.json" \
    --val_cases_file "$SPLIT/cases_val.json" \
    --meta "$meta" --log_file "$log" --seed "$s" \
    $EXTRA
done

# Aggregate: print best_val per seed.
echo "=== ${PREFIX} seed summaries ==="
for s in $(seq 1 "$N"); do
  meta="${PREFIX}_seed${s}_${stamp}"
  f="checkpoints/vessel_aware_cvae/${meta}/training_summary.json"
  [ -f "$f" ] && python -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"seed {sys.argv[2]:>2}: best_val={d['best_val_total']:.6f} @ ep{d['best_epoch']:>3} (gap={d['best_gap']:.4f})\")" "$f" "$s"
done
