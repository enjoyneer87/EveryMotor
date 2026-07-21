#!/usr/bin/env python3
"""
훈련 모니터링 웹 서버
Flask를 사용하여 RNN과 MGN 훈련 상황을 실시간으로 제공
"""

from flask import Flask, render_template_string, jsonify
import json
from pathlib import Path
from datetime import datetime
import re
import subprocess

app = Flask(__name__)

# 호스트 로컬 경로 설정
LOCAL_DATA_PATH = Path('D:/KDH/NvidiaNemo')

def get_training_status():
    """현재 훈련 상태를 반환 (도커 컨테이너 확인)"""
    try:
        result = subprocess.run(
            ['docker', 'exec', 'motor_compare', 'bash', '-c',
             'ps aux | grep "train_doe" | grep -v grep'],
            capture_output=True, text=True, timeout=5
        )
        processes = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # train_doe_meshgraphnet_aj.py was deleted (record-level split, rule 1) and
        # the RNN was retired, so matching either name would pin these False forever
        # and the monitor would report "not training" during a real run. Match the
        # trainers that actually exist.
        rnn_running = False  # RNN retired; kept so the response shape does not change
        mgn_running = any(
            'train_doe_curl_mgn.py' in p or 'train_doe_meshgraphnet.py' in p
            for p in processes
        )
    except:
        rnn_running = mgn_running = False
    
    return {
        'rnn_running': rnn_running,
        'mgn_running': mgn_running,
        'timestamp': datetime.now().isoformat()
    }

def get_log_stats(log_file):
    """로컬 로그 파일 통계 반환"""
    try:
        log_path = LOCAL_DATA_PATH / log_file
        
        if not log_path.exists():
            return {
                'lines': 0, 'epoch': 0, 'total_epochs': 0, 
                'epoch_display': 'No file', 'recent': 'Waiting for training...'
            }
        
        # 파일 라인 수
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            lines = len(all_lines)
        
        # 현재 에포크와 Loss 정보 추출
        current_epoch = 0
        total_epochs = 30  # 기본값
        recent_loss = 'N/A'
        epoch_display = 'Loading data...'
        
        for line in reversed(all_lines[-50:]):
            line_lower = line.lower()
            
            # Epoch 정보 찾기 (예: "Epoch 5/30" 또는 "[Epoch 5]")
            epoch_match = re.search(r'epoch\s*(\d+)(?:/(\d+))?', line_lower)
            if epoch_match and current_epoch == 0:
                current_epoch = int(epoch_match.group(1))
                if epoch_match.group(2):
                    total_epochs = int(epoch_match.group(2))
            
            # Loss 정보 찾기 (예: "Loss: 0.0234" 또는 "loss=0.0234")
            if not recent_loss or recent_loss == 'N/A':
                loss_match = re.search(r'(?:loss|val_loss)\s*[:=]\s*([\d.e-]+)', line_lower)
                if loss_match:
                    recent_loss = f"{float(loss_match.group(1)):.6f}"
        
        # 에포크 진행률 계산
        if current_epoch > 0 and total_epochs > 0:
            progress = (current_epoch / total_epochs) * 100
            epoch_display = f"Epoch {current_epoch}/{total_epochs} ({progress:.1f}%)"
            if recent_loss != 'N/A':
                epoch_display += f" | Loss: {recent_loss}"
        else:
            epoch_display = 'Loading data...'
        
        # 최근 진행 상황
        recent = 'Loading...'
        for line in reversed(all_lines[-20:]):
            if line.strip() and 'WARN' not in line and 'parse error' not in line.lower():
                recent = line.strip()[-120:]
                break
        
        return {
            'lines': lines,
            'epoch': current_epoch,
            'total_epochs': total_epochs,
            'epoch_display': epoch_display,
            'loss': recent_loss,
            'recent': recent
        }
    except Exception as e:
        return {
            'lines': 0, 'epoch': 0, 'total_epochs': 0,
            'epoch_display': 'Error reading log',
            'loss': 'N/A',
            'recent': str(e)
        }

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Training Monitor</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .monitor-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            }
            .card h2 {
                color: #333;
                margin-bottom: 15px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 1.5s ease-in-out infinite;
            }
            .status-indicator.running {
                background-color: #4CAF50;
            }
            .status-indicator.stopped {
                background-color: #f44336;
            }
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.5;
                }
            }
            .stat-item {
                margin: 15px 0;
                padding: 12px;
                background: #f5f5f5;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .stat-label {
                font-weight: bold;
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 5px;
            }
            .stat-value {
                font-size: 1.4em;
                color: #333;
                font-weight: bold;
                font-family: 'Courier New', monospace;
            }
            .epoch-badge {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 25px;
                font-weight: bold;
                margin-top: 10px;
                font-size: 1.1em;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
                letter-spacing: 0.5px;
            }
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
                margin-top: 10px;
                overflow: hidden;
            }
            .progress-bar-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
                transition: width 0.3s ease;
                box-shadow: 0 0 10px rgba(76, 175, 80, 0.6);
            }
            .timestamp {
                text-align: right;
                color: #999;
                font-size: 0.85em;
                margin-top: 15px;
            }
            .controls {
                text-align: center;
                margin: 20px 0;
            }
            .btn {
                background: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                margin: 0 10px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.2);
            }
            .btn-refresh {
                background: #4CAF50;
                color: white;
            }
            .btn-stop {
                background: #f44336;
                color: white;
            }
            .recent-log {
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                max-height: 100px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                color: #666;
                margin-top: 10px;
                border-left: 4px solid #667eea;
            }
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                color: #1565c0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Model Training Monitor</h1>
                <p>RNN & MGN 모델 훈련 실시간 모니터링</p>
            </div>
            
            <div class="controls">
                <button class="btn btn-refresh" onclick="updateStats()">🔄 새로고침</button>
            </div>
            
            <div class="monitor-grid">
                <!-- RNN 카드 -->
                <div class="card">
                    <h2>
                        <span class="status-indicator running" id="rnn-status-indicator"></span>
                        RNN Model
                    </h2>
                    
                    <div class="stat-item">
                        <div class="stat-label">상태</div>
                        <div class="stat-value" id="rnn-status">확인 중...</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">로그 라인 수</div>
                        <div class="stat-value" id="rnn-lines">0</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">진행 상황</div>
                        <div class="stat-value" id="rnn-epoch">데이터 로드 중...</div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill"></div>
                        </div>
                    </div>
                    
                    <div class="recent-log" id="rnn-recent">로드 중...</div>
                    <div class="timestamp" id="rnn-time">-</div>
                </div>
                
                <!-- MGN 카드 -->
                <div class="card">
                    <h2>
                        <span class="status-indicator running" id="mgn-status-indicator"></span>
                        MGN Model
                    </h2>
                    
                    <div class="stat-item">
                        <div class="stat-label">상태</div>
                        <div class="stat-value" id="mgn-status">확인 중...</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">로그 라인 수</div>
                        <div class="stat-value" id="mgn-lines">0</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">진행 상황</div>
                        <div class="stat-value" id="mgn-epoch">데이터 로드 중...</div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill"></div>
                        </div>
                    </div>
                    
                    <div class="recent-log" id="mgn-recent">로드 중...</div>
                    <div class="timestamp" id="mgn-time">-</div>
                </div>
            </div>
            
            <div class="info-box">
                <strong>ℹ️ 정보:</strong><br>
                • 자동 새로고침: 10초 주기<br>
                • 에포크진행률과 Loss 값을 실시간으로 표시합니다<br>
                • RNN & MGN 모델을 병렬로 훈련 중입니다<br>
                • 훈련이 완료되면 평가 단계로 진행됩니다
            </div>
        </div>
        
        <script>
            let autoRefreshInterval;
            
            async function updateStats() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    // RNN 상태 업데이트
                    updateModelCard('rnn', data.rnn);
                    updateModelCard('mgn', data.mgn);
                    updateStatusIndicators(data.status);
                    
                    // 타임스탐프 업데이트
                    document.getElementById('rnn-time').textContent = 
                        '마지막 업데이트: ' + new Date(data.timestamp).toLocaleTimeString('ko-KR');
                    document.getElementById('mgn-time').textContent = 
                        '마지막 업데이트: ' + new Date(data.timestamp).toLocaleTimeString('ko-KR');
                    
                } catch (error) {
                    console.error('Error fetching status:', error);
                }
            }
            
            function updateModelCard(prefix, data) {
                if (!data) return;
                
                const lines = data.lines || 0;
                const epoch = data.epoch || 0;
                const totalEpochs = data.total_epochs || 30;
                const epochDisplay = data.epoch_display || '데이터 로드 중...';
                const loss = data.loss || 'N/A';
                const recentLog = data.recent || '업데이트 대기 중...';
                
                // 라인 수 업데이트
                document.getElementById(`${prefix}-lines`).textContent = lines;
                
                // 에포크 진행 상황 업데이트 (상세 정보 포함)
                const epochDiv = document.getElementById(`${prefix}-epoch`);
                epochDiv.innerHTML = `<span class="epoch-badge">${epochDisplay}</span>`;
                
                // 진행률 바 업데이트
                let progressPercent = 0;
                if (epoch > 0 && totalEpochs > 0) {
                    progressPercent = (epoch / totalEpochs) * 100;
                }
                const progressBar = epochDiv.parentElement.querySelector('.progress-bar-fill');
                if (progressBar) {
                    progressBar.style.width = Math.max(5, progressPercent) + '%';
                    progressBar.style.animation = 'none';  // 진행률에 따라 멈추기
                }
                
                // 최근 로그 업데이트
                document.getElementById(`${prefix}-recent`).textContent = recentLog;
            }
            
            function updateStatusIndicators(status) {
                const rnnStatus = status.rnn_running ? '훈련 중 🔥' : '중단됨 ⏸️';
                const mgnStatus = status.mgn_running ? '훈련 중 🔥' : '중단됨 ⏸️';
                
                document.getElementById('rnn-status').textContent = rnnStatus;
                document.getElementById('mgn-status').textContent = mgnStatus;
                
                const rnnInd = document.getElementById('rnn-status-indicator');
                const mgnInd = document.getElementById('mgn-status-indicator');
                
                rnnInd.className = `status-indicator ${status.rnn_running ? 'running' : 'stopped'}`;
                mgnInd.className = `status-indicator ${status.mgn_running ? 'running' : 'stopped'}`;
            }
            
            // 초기 로드
            updateStats();
            
            // 10초마다 자동 새로고침
            autoRefreshInterval = setInterval(updateStats, 10000);
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/api/status')
def api_status():
    """JSON API 엔드포인트"""
    return jsonify({
        'rnn': get_log_stats('train_rnn_new.log'),
        'mgn': get_log_stats('train_mgn_aj.log'),
        'status': get_training_status(),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("훈련 모니터링 서버 시작")
    print("=" * 60)
    print("브라우저에서 다음 주소로 접속하세요:")
    print("http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
