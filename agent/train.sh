#!/bin/bash
# train.sh - 专门用于模型训练

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit
PROJECT_ROOT=$(pwd)
DATA_PATH="../data/net_data.csv"

# ------------------------------------------
# 🐍 Python Detection
# ------------------------------------------
PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    echo "❌ Error: 'python3' not found."
    exit 1
fi

echo "=================================================="
echo "   🧠 TokenFlow - Model Trainer"
echo "=================================================="
echo "📂 Working directory: $PROJECT_ROOT"
echo "🐍 Using Python: $PYTHON_EXEC"

# # 1. Check Data
# if [ ! -f "net_data.csv" ] || [ $(wc -l < net_data.csv) -lt 10 ]; then
#     echo "❌ Error: net_data.csv not found or too small."
#     echo "👉 Please run './collect_data.sh' first to generate data."
#     exit 1
# fi

# 2. Train Model
echo "🚀 Starting training..."
$PYTHON_EXEC train_model.py --data_path $DATA_PATH

if [ $? -eq 0 ]; then
    echo "✅ Training finished successfully."
    echo "👉 You can now run './run_demo.sh'"
else
    echo "❌ Training failed. Please check the python errors above."
    exit 1
fi