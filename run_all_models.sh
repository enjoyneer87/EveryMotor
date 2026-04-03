#!/bin/bash
# =============================================================
#  Motor FEA Multi-Model Training Pipeline
#  MGN vs FNO vs GINO vs Seq2SeqRNN
#
#  Docker Container: friendly_knuth (PhysicsNeMo 26.03, RTX 3090)
#
#  사전 조건:
#    - DOE H5 데이터가 /workspace/doe_data/ 에 존재
#    - MGN 학습 완료 (doe_meshgraphnet_ckpt.pt)
#
#  사용법:
#    1) Windows에서 스크립트를 Docker로 복사:
#       docker cp D:\KDH\NvidiaNemo\doe_data_utils.py friendly_knuth:/workspace/
#       docker cp D:\KDH\NvidiaNemo\train_doe_fno.py friendly_knuth:/workspace/
#       docker cp D:\KDH\NvidiaNemo\train_doe_gino.py friendly_knuth:/workspace/
#       docker cp D:\KDH\NvidiaNemo\train_doe_rnn.py friendly_knuth:/workspace/
#       docker cp D:\KDH\NvidiaNemo\compare_models.py friendly_knuth:/workspace/
#
#    2) Docker에서 실행:
#       docker exec -it friendly_knuth bash /workspace/run_all_models.sh
# =============================================================

set -e

echo "============================================"
echo "  Motor FEA Multi-Model Training Pipeline"
echo "============================================"

# --- 0. Install neuraloperator (for GINO) ---
echo ""
echo "[0/4] Installing neuraloperator library..."
pip install neuraloperator 2>/dev/null || {
    echo "  pip install failed, trying from source..."
    cd /workspace
    if [ ! -d "neuraloperator" ]; then
        git clone https://github.com/neuraloperator/neuraloperator.git
    fi
    cd neuraloperator
    pip install -e . 2>/dev/null
    cd /workspace
}
echo "  neuraloperator installed."

# --- 1. FNO Training ---
echo ""
echo "============================================"
echo "[1/4] Training FNO (Fourier Neural Operator)"
echo "============================================"
python /workspace/train_doe_fno.py \
    --data-dir /workspace/doe_data \
    --grid-res 64 \
    --epochs 60 \
    --batch-size 8 \
    --lr 1e-3 \
    --fno-modes 16 \
    --fno-layers 4 \
    --fno-hidden 64 \
    --ckpt /workspace/doe_fno_ckpt.pt \
    2>&1 | tee /workspace/train_fno.log

# --- 2. GINO Training ---
echo ""
echo "============================================"
echo "[2/4] Training GINO (Geometry-Informed NO)"
echo "============================================"
python /workspace/train_doe_gino.py \
    --data-dir /workspace/doe_data \
    --latent-res 32 \
    --epochs 60 \
    --lr 1e-3 \
    --fno-modes 16 \
    --fno-layers 4 \
    --fno-hidden 64 \
    --gno-radius 0.1 \
    --ckpt /workspace/doe_gino_ckpt.pt \
    2>&1 | tee /workspace/train_gino.log

# --- 3. Seq2SeqRNN Training ---
echo ""
echo "============================================"
echo "[3/4] Training Seq2SeqRNN (ConvGRU)"
echo "============================================"
python /workspace/train_doe_rnn.py \
    --data-dir /workspace/doe_data \
    --grid-res 64 \
    --seq-in 4 \
    --seq-out 4 \
    --epochs 60 \
    --batch-size 4 \
    --lr 1e-3 \
    --hidden-channels 64 \
    --ckpt /workspace/doe_rnn_ckpt.pt \
    2>&1 | tee /workspace/train_rnn.log

# --- 4. Comparison ---
echo ""
echo "============================================"
echo "[4/4] Model Comparison"
echo "============================================"
python /workspace/compare_models.py \
    --data-dir /workspace/doe_data \
    --mgn-ckpt /workspace/doe_meshgraphnet_ckpt.pt \
    --fno-ckpt /workspace/doe_fno_ckpt.pt \
    --gino-ckpt /workspace/doe_gino_ckpt.pt \
    --rnn-ckpt /workspace/doe_rnn_ckpt.pt \
    --out-dir /workspace/model_comparison \
    2>&1 | tee /workspace/comparison.log

echo ""
echo "============================================"
echo "  All training complete!"
echo "  Checkpoints:"
echo "    MGN:  /workspace/doe_meshgraphnet_ckpt.pt"
echo "    FNO:  /workspace/doe_fno_ckpt.pt"
echo "    GINO: /workspace/doe_gino_ckpt.pt"
echo "    RNN:  /workspace/doe_rnn_ckpt.pt"
echo "  Comparison: /workspace/model_comparison/"
echo "============================================"
