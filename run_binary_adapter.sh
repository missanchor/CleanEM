#!/bin/bash
# Binary Adapter Training Script with GPU Support
# Usage: ./run_binary_adapter.sh

set -e  # Exit on error

echo "================================================"
echo "  Qwen2.5-0.5B Binary Adapter Training"
echo "================================================"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "
import torch
gpu_count = torch.cuda.device_count()
print(f'  ✅ {gpu_count} GPU(s) detected')
if gpu_count > 0:
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        print(f'     GPU {i}: {name}')
    print('  🚀 GPU mode will be used')
else:
    print('  ⚠️  No GPU detected, using CPU')
"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU by default
CACHE_DIR="/data/nw/modelscope_cache"
LOCAL_DIR=""

# Check if cache exists
if [ -d "$CACHE_DIR" ] && [ "$(ls -A $CACHE_DIR 2>/dev/null | grep -v '^.lock$')" ]; then
    echo "✅ Cache directory exists and contains model files"
    echo "📁 Cache location: $CACHE_DIR"
    echo ""
    echo "🚀 Starting training with cached model..."
else
    echo "⚠️  Cache directory is empty or doesn't exist"
    echo "📥 Will download model on first run..."
    echo ""
fi

# Run the training script
/mnt/data/welkinni/anaconda3/envs/cleaningllm/bin/python /data/nw/Cleaning_LLM/modelscope_adapter_binary_demo.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --cache-dir "$CACHE_DIR" \
    ${LOCAL_DIR:+--local-dir "$LOCAL_DIR"} \
    --epochs 1 \
    --batch-size 2 \
    --lr 5e-4

echo ""
echo "✅ Training completed!"
