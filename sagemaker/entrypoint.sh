#!/bin/bash
# SageMaker entry point for Memory-R1 Workshop experiments.
# LoRA on 4x A10G. Supports budget-constrained forgetting experiments.
set -euo pipefail

echo "============================================"
echo "Memory-R1 Workshop Experiments"
echo "============================================"
echo "Phase: ${SM_HP_PHASE:-mm}"
echo "Budget λ: ${SM_HP_BUDGET_LAMBDA:-0.0}"
echo "Reward: ${SM_HP_REWARD:-em}"
echo "SFT warmstart: ${SM_HP_SFT_WARMSTART:-false}"
echo "Max steps: ${SM_HP_MAX_STEPS:-200}"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
echo "GPUs: $NUM_GPUS"
nvidia-smi 2>/dev/null || true
echo "============================================"

INPUT_DIR="${SM_CHANNEL_TRAINING:-/opt/ml/input/data/training}"
MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/model}"
PHASE="${SM_HP_PHASE:-mm}"
MAX_STEPS="${SM_HP_MAX_STEPS:-200}"
EVAL_EVERY="${SM_HP_EVAL_EVERY:-20}"
CHECKPOINT_EVERY="${SM_HP_CHECKPOINT_EVERY:-50}"
BASE_MODEL="${SM_HP_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BUDGET_LAMBDA="${SM_HP_BUDGET_LAMBDA:-0.0}"
BUDGET_TARGET="${SM_HP_BUDGET_TARGET:-50}"
REWARD="${SM_HP_REWARD:-em}"
SFT_WARMSTART="${SM_HP_SFT_WARMSTART:-false}"
ORDER="${SM_HP_ORDER:-mm-aa}"

export HF_HOME="/tmp/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$MODEL_DIR"

# ---- Extract code ----
if [ -f "$INPUT_DIR/agents-memory-v2.tar" ]; then
    echo "Extracting code..."
    tar -xf "$INPUT_DIR/agents-memory-v2.tar" -C /tmp/
    PROJECT_DIR="/tmp/agents-memory"
elif [ -d "/tmp/agents-memory" ]; then
    PROJECT_DIR="/tmp/agents-memory"
else
    PROJECT_DIR="$INPUT_DIR"
fi

# ---- Pre-cached model weights ----
PREP_DIR="${SM_CHANNEL_PREP:-}"
if [ -n "$PREP_DIR" ]; then
    if [ -f "$PREP_DIR/model.tar.gz" ]; then
        echo "Extracting model weights..."
        mkdir -p /tmp/prep_extracted
        tar -xzf "$PREP_DIR/model.tar.gz" -C /tmp/prep_extracted/
        PREP_DIR="/tmp/prep_extracted"
    fi
    if [ -d "$PREP_DIR/base_model" ]; then
        BASE_MODEL="$PREP_DIR/base_model"
        echo "Using cached model: $BASE_MODEL"
    fi
fi

# ---- Previous job outputs (adapter chaining) ----
PREV_DIR="${SM_CHANNEL_PREV:-}"
if [ -n "$PREV_DIR" ]; then
    if [ -f "$PREV_DIR/model.tar.gz" ]; then
        echo "Extracting previous outputs..."
        mkdir -p /tmp/prev_extracted
        tar -xzf "$PREV_DIR/model.tar.gz" -C /tmp/prev_extracted/
        PREV_DIR="/tmp/prev_extracted"
    fi
    if [ -d "$PREV_DIR" ]; then
        mkdir -p "$PROJECT_DIR/models"
        cp -r "$PREV_DIR"/* "$PROJECT_DIR/models/" 2>/dev/null || true
        echo "Previous adapters loaded."
    fi
fi

cd "$PROJECT_DIR"

# ---- Install uv + Python 3.13 ----
echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Python 3.13 venv ==="
uv venv --python 3.13 .venv
source .venv/bin/activate

echo "=== Dependencies ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install -e ".[training]"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# ---- Check data ----
DATA_DIR="$PROJECT_DIR/data/r1_training"
if [ ! -f "$DATA_DIR/answer_agent_train.jsonl" ]; then
    echo "ERROR: Training data not found"
    exit 1
fi
echo "Data found."

# ---- Build training command ----
WARMSTART_FLAG=""
if [ "$SFT_WARMSTART" = "true" ]; then
    WARMSTART_FLAG="--sft-warmstart"
fi

# Find frozen AA adapter for MM training
FROZEN_AA_FLAG=""
for p in "$PROJECT_DIR/models/memory-r1-rl/adapter_answer_agent_rl/best" \
         "$PROJECT_DIR/models/memory-r1-rl/adapter_answer_agent_rl/final"; do
    if [ -d "$p" ]; then
        FROZEN_AA_FLAG="--frozen-aa-path $p"
        echo "Frozen AA: $p"
        break
    fi
done

# ---- Run training ----
echo ""
echo "============================================"
echo "Running: phase=$PHASE reward=$REWARD budget_λ=$BUDGET_LAMBDA"
echo "============================================"

python -u scripts/train_memory_r1_rl_tracked.py \
    --phase "$PHASE" \
    --base-model "$BASE_MODEL" \
    --max-steps "$MAX_STEPS" \
    --eval-every "$EVAL_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --reward "$REWARD" \
    --budget-lambda "$BUDGET_LAMBDA" \
    --budget-target "$BUDGET_TARGET" \
    --order "$ORDER" \
    $WARMSTART_FLAG \
    $FROZEN_AA_FLAG

# ---- Copy outputs ----
echo ""
echo "Copying outputs to $MODEL_DIR"
if [ -d "$PROJECT_DIR/models" ]; then
    cp -r "$PROJECT_DIR/models/"* "$MODEL_DIR/" 2>/dev/null || true
fi
find "$PROJECT_DIR" -name "metrics.jsonl" -exec cp {} "$MODEL_DIR/" \; 2>/dev/null || true

echo "Output contents:"
find "$MODEL_DIR" -type f | head -20

echo ""
echo "============================================"
echo "COMPLETE"
echo "============================================"
