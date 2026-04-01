import sys
from pathlib import Path
import numpy as np

# pyMCAD 경로 연결
BASE_DIR = Path(r"D:/KDH/NvidiaNemo")
EMACH_PATH = BASE_DIR.parent / "gitEmach" / "eMach"
if str(EMACH_PATH) not in sys.path:
    sys.path.insert(0, str(EMACH_PATH))

try:
    from tools.motorCAD.pyMCAD.magnetic import (
        MagneticRegionsTimeSeries,
        MagneticRegions,
        MagneticRegion,
        MagElement,
    )
except ImportError as e:
    print(f"Warning: Cannot import pyMCAD tools ({e})")

def convert_nemo_to_pymcad_ts(node_x_all, node_y_all, mesh_triangles, pred_all, reg_codes=None, region_names_map=None):
    """
    모터 치수/형상(메쉬)별 NeMo 모델의 추론 결과를 pyMCAD가 사용하는
    MagneticRegionsTimeSeries 구조로 변환해 GUI Plot과 100% 호환되게 합니다.

    Args:
        node_x_all   : (n_steps, n_nodes) 배열. 각 스텝별 노드 X 좌표
        node_y_all   : (n_steps, n_nodes) 배열. 각 스텝별 노드 Y 좌표
        mesh_triangles : (n_elements, 3) 삼각 메쉬 연결 구조
        pred_all     : (n_steps, n_nodes, 4) NeMo 예측값 배열. [Bx, By, A, J]
        reg_codes    : (n_elements,) 리전 코드 옵션. (None이면 전부 1)
        
    Returns:
        ts : pyMCAD.magnetic.MagneticRegionsTimeSeries 객체 (GUI 플롯 호환)
    """
    ts = MagneticRegionsTimeSeries()
    n_steps = len(node_x_all)
    n_elements = len(mesh_triangles)
    
    if reg_codes is None:
        reg_codes = np.ones(n_elements, dtype=np.int32)
        
    for step in range(n_steps):
        regions_step = MagneticRegions()
        
        # 1. 노드 좌표계 세팅 (NodeIndex -> (x, y))
        node_xy_map = {}
        for nid, (nx, ny) in enumerate(zip(node_x_all[step], node_y_all[step])):
            node_xy_map[nid] = (nx, ny)
        regions_step.set_node_xy(node_xy_map)
        
        # 2. 리전 생성 준비
        max_reg = int(np.max(reg_codes))
        regions_step.ensure_region(max_reg)

        if region_names_map:
            for rc, rname in region_names_map.items():
                if rc - 1 < len(regions_step):
                    regions_step[rc - 1].region_name = str(rname)
                    regions_step[rc - 1].reg_code = rc

        
        # 3. Element 데이터 매핑 (Node 예측값을 Element 대푯값으로 변환)
        for e_idx in range(n_elements):
            n1, n2, n3 = mesh_triangles[e_idx]
            
            if max(n1, n2, n3) >= len(node_x_all[step]):
                continue
                
            rc = reg_codes[e_idx]
            
            # [Bx, By, A, J]
            v1 = pred_all[step][n1]
            v2 = pred_all[step][n2]
            v3 = pred_all[step][n3]
            
            # 요소 중심(Centroid) 기준 평균값 도출
            e_bx = (v1[0] + v2[0] + v3[0]) / 3.0
            e_by = (v1[1] + v2[1] + v3[1]) / 3.0
            e_a  = (v1[2] + v2[2] + v3[2]) / 3.0
            e_j  = (v1[3] + v2[3] + v3[3]) / 3.0
            
            el = MagElement(
                tri_index=e_idx+1,
                node_1=n1,
                node_2=n2,
                node_3=n3,
                reg_code=rc,
                bx=e_bx,
                by=e_by,
                a=e_a,
                j=e_j
            )
            # reg_code - 1 이 region 인덱스
            regions_step[rc - 1].elements.append(el)
            
        ts.by_step[step] = regions_step
        
    return ts
