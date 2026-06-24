# NvidiaNemo Python 코드의 MATLAB 호출 분석 결과

**검색 날짜**: 2026년 4월 6일  
**검색 범위**: `d:\KDH\NvidiaNemo\` 및 하위 모든 `.py` 파일  
**DOE 데이터 경로**: `D:\KDH\Sim_4SolverX\DOE_TrainingData\case_NNNN\postproc\`

---

## 📋 요약

NvidiaNemo 워크스페이스의 **Python 코드에서 MATLAB을 직접 호출하는 코드는 없습니다**.

### 아키텍처 구조:
```
MATLAB (dev4ThesisDOE.mlx 등)
    ↓ 생성
H5 파일 (Mag_OnLoadTorque_result_1.h5 등)
    ↓ 로드
Python (h5py 사용)
    ↓ 처리/학습
Neural Network (PhysicsNeMo, FNO, GINO, RNN)
```

---

## 🔍 발견 사항

### 1️⃣ MATLAB 라이브 스크립트 파일 (.mlx)

**위치**: `d:\KDH\NvidiaNemo\eMach\mlxperPJT\MCADLab\`

| 파일명 | 용도 | 상태 |
|--------|------|------|
| **dev4ThesisDOE.mlx** | DOE 생성 (MATLAB 기반) | 실행 - Python 호출 없음 |
| **dev4ThesisDOE_KDHPC.mlx** | DOE 생성 (KDHPC 서버 버전) | 실행 - Python 호출 없음 |
| devCompareGetMotoLabParameterData.mlx | MotorLAB 데이터 비교 | 실행 - Python 호출 없음 |
| devImportLab4MCADHybridACLossMethod.mlx | AC 손실 계산 | 실행 - Python 호출 없음 |
| devdoForScaledResult4HDEV.mlx | 스케일 결과 처리 | 실행 - Python 호출 없음 |
| Result4Paper.mlx | 논문 결과 시각화 | 실행 - Python 호출 없음 |

**결론**: 이 .mlx 파일들은 **MATLAB 사용자가 수동으로 실행**하여 DOE 데이터를 생성합니다.

---

### 2️⃣ Python에서 예상되는 MATLAB 호출 코드들

#### ❌ 찾지 못한 것들:
- `subprocess.Popen(['matlab', ...])`
- `subprocess.run(['matlab.exe', ...])`
- `matlab.engine.start_matlab()`
- `pymatlab` 모듈 사용
- `.m` 또는 `.mlx` 파일 실행 코드
- `os.system("matlab ...")` 호출
- `execfile` 또는 `exec()` 기반의 MATLAB 스크립트 실행

#### ⚠️ 참고: zipfile 사용 (하지만 MATLAB 호출 아님)
[download_gdrive_requests.py](download_gdrive_requests.py#L138)에서 zipfile을 사용하지만, 이는:
- Google Drive에서 다운로드한 파일 추출용
- MATLAB .mlx 파일의 CDATA 추출용이 아님

---

### 3️⃣ DOE 데이터 로딩 구조 (Python 코드)

Python은 **이미 생성된 DOE 데이터를 읽기만 함**:

#### 핵심 파일들:

| 파일 | 기능 | 데이터 로드 방식 |
|------|------|-----------------|
| [doe_data_utils.py](doe_data_utils.py) | DOE 데이터 유틸리티 | `h5py.File()` |
| [convert_doe_to_fsg.py](convert_doe_to_fsg.py) | H5 → FSG 변환 | `h5py.File()` |
| [phase1_static/motor_dataset.py](phase1_static/motor_dataset.py#L419) | 데이터셋 로더 | `h5py.File()` |
| [infer_all_steps_nodes.py](infer_all_steps_nodes.py#L92) | 추론 스크립트 | `h5py.File()` |

#### 로드 경로 패턴:
```python
# 예시: phase1_static/motor_dataset.py
h5_file = data_dir / f"case_{case['index']:04d}" / "postproc" / h5_basename
h5_data = h5py.File(h5_file, 'r')
```

#### H5 파일 분류:
```python
# doe_data_utils.py의 classify_motorcad_h5_filename()
- "OnLoadTorque"           → OnLoadTorque (시간 기반, 시뮬레이션 결과)
- "LossElement"/"OnLoadLoss" → LossElement_OnLoadLoss (손실 데이터)
- "StaticLoadInductance"   → StaticLoadInductance (정적 인덕턴스)
- "StaticLoad"             → StaticLoad (정적 하중)
- "StaticOC"               → StaticOC (정적 오픈 회로)
```

---

### 4️⃣ subprocess 호출 분석

**subprocess 사용 위치들** (MATLAB 호출 아님):

| 파일 | 라인 | 용도 |
|------|------|------|
| [check_submodule_branch.py](check_submodule_branch.py#L13) | 13 | Git 서브모듈 상태 확인 |
| [check_gino_status.py](check_gino_status.py#L2) | 2 | Docker 컨테이너 상태 확인 |
| [check_jupyter.py](check_jupyter.py#L4) | 4 | Jupyter 상태 확인 |
| [check_npz.py](check_npz.py#L2) | 2 | Docker Python 실행 |
| [check_scatter.py](check_scatter.py#L3) | 3 | Docker 명령 실행 |
| [check_status_v2.py](check_status_v2.py#L4) | 4 | Docker/Process 상태 확인 |
| [download_motor_drive.py](download_motor_drive.py#L90) | 90 | 파일 다운로드 |
| [tests/test_overfit_single_smoke.py](tests/test_overfit_single_smoke.py#L57) | 57 | Docker 내 학습 실행 |

**결론**: 모두 Docker 또는 Git 관련 호출임.

---

### 5️⃣ eMach 서브모듈의 MATLAB 관련 코드

#### eMach 내 MATLAB 코드 (주석 형태):
- [eMach/Class/pyMotorGeo/face_detection.py](eMach/Class/pyMotorGeo/face_detection.py#L9) — MATLAB BanGeoCode를 Python으로 포팅
- [eMach/tools/motorCAD/pyMCAD/melec_req_check.py](eMach/tools/motorCAD/pyMCAD/melec_req_check.py#L876) — `plotNrunMCADLab()` 함수 (MotorLAB MAT 파일 처리)

**함수 `plotNrunMCADLab()`**:
- MotorCAD의 자기 해석 계산 실행 (`mc.calculate_magnetic_lab()`)
- 결과 MAT 파일 로드
- Efficiency 맵 시각화

---

## 📍 DOE 데이터 생성 흐름

### MATLAB 사계 (수동)
```
1. 사용자가 dev4ThesisDOE.mlx 열기
   ↓
2. MATLAB에서 DOE 케이스 생성
   - 파라미터 설정 (Ratio_Bore, PeakCurrent, PhaseAdvance 등)
   ↓
3. MotorCAD FEA 시뮬레이션 실행
   ↓
4. 결과를 H5 파일로 저장
   - Mag_OnLoadTorque_result_1.h5
   - Mag_StaticLoad_result_1.h5
   - 등...
   ↓
5. 경로: D:\KDH\Sim_4SolverX\DOE_TrainingData\case_NNNN\postproc\
```

### Python 사계 (자동)
```
1. load_doe_data() 호출
   ↓
2. doe_manifest.json 파일 읽기
   ↓
3. H5 파일 경로 구성
   - case_{index:04d}/postproc/{filename}.h5
   ↓
4. h5py로 데이터 로드
   - mesh/node_id, node_x_mm, node_y_mm
   - fields/bx, by, a, j
   - steps, meta/time_s 등
   ↓
5. 신경망 훈련 (PhysicsNeMo, FNO, GINO, RNN)
```

---

## ⚙️ 핵심 데이터 로딩 함수

### [doe_data_utils.py](doe_data_utils.py#L47)

```python
def parse_h5_timeseries(path: Path, max_steps: Optional[int] = None) -> List[dict]:
    """Parse a pyMCAD magnetic timeseries H5 file into record dicts."""
    with h5py.File(path, "r") as f:
        steps = np.asarray(f["steps"][:], dtype=np.int32)
        node_id = np.asarray(f["mesh/node_id"][:], dtype=np.int32)
        node_x0 = np.asarray(f["mesh/node_x_mm"][:], dtype=np.float64)
        node_y0 = np.asarray(f["mesh/node_y_mm"][:], dtype=np.float64)
        
        bx = np.asarray(f["fields/bx"][:], dtype=np.float32)
        by = np.asarray(f["fields/by"][:], dtype=np.float32)
        a = np.asarray(f["fields/a"][:], dtype=np.float32)
        j = np.asarray(f["fields/j"][:], dtype=np.float32)
        # ...
        return records
```

### [doe_data_utils.py](doe_data_utils.py#L400)

```python
def load_doe_data(data_dir: str, max_steps_per_case: Optional[int] = None
                  ) -> Tuple[List[dict], List[Dict[str, float]]]:
    """Load all DOE cases from manifest."""
    manifest_path = data_dir / "doe_manifest.json"
    
    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        for h5p in h5_paths:
            candidates = [
                Path(h5p),
                data_dir / f"case_{case['index']:04d}" / "postproc" / h5_basename,
                data_dir / h5_basename,
            ]
            h5_file = next((c for c in candidates if c.exists()), None)
            if h5_file:
                records = parse_h5_timeseries(h5_file, max_steps=max_steps_per_case)
```

---

## 🎯 결론

### ✅ 확인된 사실

1. **MATLAB 호출 없음** — Python은 이미 생성된 DOE 데이터만 로드
2. **.mlx 파일은 MATLAB에서 수동 실행** — Python 자동화 불가
3. **H5 기반 데이터 파이프라인** — h5py로 읽음 → 신경망 학습
4. **manifest.json** — DOE 케이스/경로 메타데이터 관리
5. **dev4ThesisDOE.mlx** — MATLAB 사용자가 DOE 데이터 생성 도구

### 💡 권장 사항

DOE를 **자동으로 생성해야 한다면**:
1. MATLAB to Python 포팅 (고비용)
2. MATLAB Compiler를 통한 실행파일화 
3. MATLAB Engine API 활용 (Windows + MATLAB 설치 필수)
4. 클라우드 MATLAB (MathWorks) 활용

현재 구조는 **MATLAB에서 데이터 생성 → Python에서 학습**이라는 명확한 분리로 설계되어 있습니다.

---

## 📚 참고 파일 목록

### Python DOE 데이터 로딩
- [doe_data_utils.py](doe_data_utils.py)
- [doe_data_utils_check.py](doe_data_utils_check.py)
- [convert_doe_to_fsg.py](convert_doe_to_fsg.py)
- [phase1_static/motor_dataset.py](phase1_static/motor_dataset.py)

### 추론 스크립트
- [infer_all_steps_nodes.py](infer_all_steps_nodes.py)
- [infer_doe_meshgraphnet.py](infer_doe_meshgraphnet.py)
- [infer_phase1_pbc.py](infer_phase1_pbc.py)

### MATLAB 도구 (Python 래퍼)
- [eMach/tools/motorCAD/pyMCAD/](eMach/tools/motorCAD/pyMCAD/) — pyMotorCAD 바인딩
- [eMach/Class/pyMotorGeo/motorcad_bridge.py](eMach/Class/pyMotorGeo/motorcad_bridge.py)

### MATLAB 라이브 스크립트
- [eMach/mlxperPJT/MCADLab/dev4ThesisDOE.mlx](eMach/mlxperPJT/MCADLab/dev4ThesisDOE.mlx)
- [eMach/mlxperPJT/MCADLab/dev4ThesisDOE_KDHPC.mlx](eMach/mlxperPJT/MCADLab/dev4ThesisDOE_KDHPC.mlx)
