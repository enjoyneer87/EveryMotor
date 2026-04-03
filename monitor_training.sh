#!/bin/bash
echo '모델 훈련 진행 사항 모니터...'
while true; do
    clear
    echo '=== FNO 훈련 ==='
    tail -1 /workspace/host_data/train_fno_new.log 2>/dev/null || echo '준비 중...'
    echo ''
    echo '=== GINO 훈련 ===' 
    tail -1 /workspace/host_data/train_gino_new.log 2>/dev/null || echo '준비 중...'
    echo ''
    echo '=== RNN 훈련 ==='
    tail -1 /workspace/host_data/train_rnn_new.log 2>/dev/null || echo '준비 중...'
    echo ''
    echo '진행 중...'
    ps aux | grep 'train_doe' | grep -v grep | wc -l
    sleep 10
done
