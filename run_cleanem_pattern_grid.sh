#!/usr/bin/env bash
set -u

ROOT=/home/xkh/project/dataclean/table_det
PY=/home/xkh/miniconda3/bin/conda
BASE_URL=http://127.0.0.1:8003/v1
MODEL=qwen3-32b
DATASET=$1

case "$DATASET" in
  beers)
    DIRTY=data/beers_error-01.csv
    CLEAN=data/beers_clean.csv
    RULE_CACHE=results/commit2_gate_ablation/cache/beers_rule_pool.json
    ACTIVE_CACHE=results/label_budget_ablation/budget10/beers/cache/d_optimal_with_gate_v4.json
    PLAN_CACHE=results/agent_pattern_weight_grid/cache/beers_canonicalization_plans.json
    ;;
  hospital)
    DIRTY=data/hospital_error-01.csv
    CLEAN=data/hospital_clean.csv
    RULE_CACHE=results/commit3_generalization/hospital/cache/rule_pool.json
    ACTIVE_CACHE=results/label_budget_ablation/budget10/hospital/cache/d_optimal_with_gate_v4.json
    PLAN_CACHE=results/agent_pattern_weight_grid/cache/hospital_canonicalization_plans.json
    ;;
  flights)
    DIRTY=data/flights_error-01.csv
    CLEAN=data/flights_clean.csv
    RULE_CACHE=results/weak_prior_gate/flights/cache/rule_pool.json
    ACTIVE_CACHE=results/weak_prior_gate/flights/cache/d_optimal_v4.json
    PLAN_CACHE=results/weak_prior_gate/flights/cache/canonicalization_plans.json
    ;;
  *)
    echo "unknown dataset: $DATASET" >&2
    exit 2
    ;;
esac

cd "$ROOT" || exit 2
for SCALE in 0.5 1.0 1.5 2.0; do
  TAG=${SCALE/./p}
  OUT=results/agent_pattern_weight_grid/canonicalization_on/${DATASET}/scale_${TAG}
  mkdir -p "$OUT"
  echo "START dataset=$DATASET scale=$SCALE time=$(date --iso-8601=seconds)"
  "$PY" run -n cleaningllm python main.py \
    --dirty_csv "$DIRTY" \
    --clean_csv "$CLEAN" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --rule_pool_cache "$RULE_CACHE" \
    --active_query_cache "$ACTIVE_CACHE" \
    --canonicalization_plan_cache "$PLAN_CACHE" \
    --active_label_budget 10 \
    --active_query_strategy d_optimal \
    --evidence_gating_mode weak_prior \
    --gate_prior_strength 0.25 \
    --gate_prior_pseudocount 2.0 \
    --agent_pattern_rule_scale "$SCALE" \
    --output_dir "$OUT"
  STATUS=$?
  echo "END dataset=$DATASET scale=$SCALE status=$STATUS time=$(date --iso-8601=seconds)"
  if [ "$STATUS" -ne 0 ]; then
    exit "$STATUS"
  fi
done
