#!/bin/bash

# 运行所有两阶段对比学习实验
# 使用指定的Python解释器和配置文件，并将输出保存到日志文件

PYTHON_BIN="/mnt/data/welkinni/table_det/.conda/envs/cleaningllm/bin/python"
MAIN_SCRIPT="/mnt/data/welkinni/table_det/main_cross_modal_detector.py"
BASE_DIR="/mnt/data/welkinni/table_det"

# 切换到项目根目录
cd "$BASE_DIR" || exit 1

# 创建logs目录（如果不存在）
mkdir -p logs

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +%Y%m%d_%H%M)

# 运行beers实验（后台并行）
echo "开始运行 beers_two_stage_experiment..."
BEERS_LOG="logs/${TIMESTAMP}_beers_two_stage_experiment.log"
"$PYTHON_BIN" "$MAIN_SCRIPT" --config configs/beers_two_stage_experiment.json > "$BEERS_LOG" 2>&1 &
BEERS_PID=$!
sleep 30

# 运行flights实验（后台并行）
echo "开始运行 flights_two_stage_experiment..."
FLIGHTS_LOG="logs/${TIMESTAMP}_flights_two_stage_experiment.log"
"$PYTHON_BIN" "$MAIN_SCRIPT" --config configs/flights_two_stage_experiment.json > "$FLIGHTS_LOG" 2>&1 &
FLIGHTS_PID=$!
sleep 30

# 运行hospital实验（后台并行）
echo "开始运行 hospital_two_stage_experiment..."
HOSPITAL_LOG="logs/${TIMESTAMP}_hospital_two_stage_experiment.log"
"$PYTHON_BIN" "$MAIN_SCRIPT" --config configs/hospital_two_stage_experiment.json > "$HOSPITAL_LOG" 2>&1 &
HOSPITAL_PID=$!
sleep 30

# 等待所有后台任务完成
echo "所有实验已在后台启动，等待完成..."
wait $BEERS_PID
echo "beers实验完成，日志保存在: $BEERS_LOG"

wait $FLIGHTS_PID
echo "flights实验完成，日志保存在: $FLIGHTS_LOG"

wait $HOSPITAL_PID
echo "hospital实验完成，日志保存在: $HOSPITAL_LOG"

echo "所有实验运行完成！"

