#!/bin/bash

# 运行所有MCM实验（Masked Cell Modeling）
# 使用指定的Python解释器和配置文件，并将输出保存到日志文件
#
# 注意：
# - 配置文件中的 experiment="contrastive_two_stage" 和 objective="mcm" 会自动调用 run_mcm_experiment
# - MCM和Two-Stage方案已拆分：MCM使用独立的 run_mcm_experiment 函数
# - 如果配置文件中包含 evaluation 部分（clean_data_path），
#   训练完成后会自动进行MCM评估，计算错误检测的 Precision、Recall 和 F1 Score
# - 评估结果会输出到日志文件中，查找 "MCM 评估结果" 部分
# - 重要：当前 generation_backend="vllm" 是“进程内 import vllm”，每个实验进程会各自加载一套 vLLM 引擎，
#   默认不建议并行跑多个实验（否则容易显存 OOM）。本脚本默认改为串行执行。

PYTHON_BIN="/mnt/data/welkinni/table_det/.conda/envs/cleaningllm/bin/python"
MAIN_SCRIPT="/mnt/data/welkinni/table_det/main_cross_modal_detector.py"
BASE_DIR="/mnt/data/welkinni/table_det"

# 配置文件（使用绝对路径，避免依赖 cwd）
BEERS_CFG="$BASE_DIR/configs/beers_mcm_experiment.json"
FLIGHTS_CFG="$BASE_DIR/configs/flights_mcm_experiment.json"
HOSPITAL_CFG="$BASE_DIR/configs/hospital_mcm_experiment.json"

# 运行模式：sequential | parallel
# - sequential: 串行跑 3 个实验（推荐，适配本地 vLLM）
# - parallel:   并行跑（仅建议多 GPU 并手动设置 CUDA_VISIBLE_DEVICES 映射）
RUN_MODE="${RUN_MODE:-parallel}"

# 切换到项目根目录
cd "$BASE_DIR" || exit 1

# 创建logs目录（如果不存在）
mkdir -p logs

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "=========================================="
echo "开始运行所有实验（MCM）"
echo "时间戳: $TIMESTAMP"
echo "RUN_MODE: $RUN_MODE"
echo "=========================================="

BEERS_LOG="logs/${TIMESTAMP}_beers_mcm_experiment.log"
FLIGHTS_LOG="logs/${TIMESTAMP}_flights_mcm_experiment.log"
HOSPITAL_LOG="logs/${TIMESTAMP}_hospital_mcm_experiment.log"

run_one () {
  local name="$1"
  local cfg="$2"
  local log="$3"
  echo "[$(date +%H:%M:%S)] 开始运行 ${name} MCM实验..."
  echo "  config: $cfg"
  echo "  log:    $log"
  "$PYTHON_BIN" "$MAIN_SCRIPT" --config "$cfg" > "$log" 2>&1
  return $?
}

BEERS_EXIT=0
FLIGHTS_EXIT=0
HOSPITAL_EXIT=0

if [ "$RUN_MODE" = "parallel" ]; then
  echo "⚠️  RUN_MODE=parallel：仅建议多 GPU 时使用，并手动设置 CUDA_VISIBLE_DEVICES 映射。"

  CUDA_VISIBLE_DEVICES=0 run_one "beers" "$BEERS_CFG" "$BEERS_LOG" &
  BEERS_PID=$!
  CUDA_VISIBLE_DEVICES=1 run_one "flights" "$FLIGHTS_CFG" "$FLIGHTS_LOG" &
  FLIGHTS_PID=$!
  CUDA_VISIBLE_DEVICES=2 run_one "hospital" "$HOSPITAL_CFG" "$HOSPITAL_LOG" &
  HOSPITAL_PID=$!

  wait $BEERS_PID; BEERS_EXIT=$?
  wait $FLIGHTS_PID; FLIGHTS_EXIT=$?
  wait $HOSPITAL_PID; HOSPITAL_EXIT=$?
else
  run_one "beers" "$BEERS_CFG" "$BEERS_LOG"; BEERS_EXIT=$?
  run_one "flights" "$FLIGHTS_CFG" "$FLIGHTS_LOG"; FLIGHTS_EXIT=$?
  run_one "hospital" "$HOSPITAL_CFG" "$HOSPITAL_LOG"; HOSPITAL_EXIT=$?
fi

if [ $BEERS_EXIT -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] ✓ beers实验完成，日志保存在: $BEERS_LOG"
  echo "  查看MCM评估结果: grep 'MCM 评估结果' $BEERS_LOG"
else
  echo "[$(date +%H:%M:%S)] ✗ beers实验失败 (退出码: $BEERS_EXIT)，日志: $BEERS_LOG"
fi

if [ $FLIGHTS_EXIT -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] ✓ flights实验完成，日志保存在: $FLIGHTS_LOG"
  echo "  查看MCM评估结果: grep 'MCM 评估结果' $FLIGHTS_LOG"
else
  echo "[$(date +%H:%M:%S)] ✗ flights实验失败 (退出码: $FLIGHTS_EXIT)，日志: $FLIGHTS_LOG"
fi

if [ $HOSPITAL_EXIT -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] ✓ hospital实验完成，日志保存在: $HOSPITAL_LOG"
  echo "  查看MCM评估结果: grep 'MCM 评估结果' $HOSPITAL_LOG"
else
  echo "[$(date +%H:%M:%S)] ✗ hospital实验失败 (退出码: $HOSPITAL_EXIT)，日志: $HOSPITAL_LOG"
fi

echo ""
echo "=========================================="
if [ $BEERS_EXIT -eq 0 ] && [ $FLIGHTS_EXIT -eq 0 ] && [ $HOSPITAL_EXIT -eq 0 ]; then
    echo "✓ 所有实验运行完成！"
    echo ""
    echo "快速查看所有MCM评估结果："
    echo "  grep -A 10 'MCM 评估结果' logs/${TIMESTAMP}_*.log"
else
    echo "⚠️  部分实验失败，请检查日志文件"
fi
echo "=========================================="

