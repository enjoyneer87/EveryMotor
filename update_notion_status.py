import urllib.request
import json
import os

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = os.getenv("NOTION_PAGE_ID", "33507031978c80118c58ca8591142be3")

if not NOTION_TOKEN:
    raise RuntimeError("NOTION_TOKEN environment variable is required.")

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

data = {
    "children": [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "🤖 Agent Update: 4SolverX 모델 통합 및 시각화 파이프라인 (Server 3389)"}}]
            }
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": "로컬 Windows 환경과 도커 컨테이너를 분리하여 다음과 같이 작업을 진행 중입니다:\n\n"}}
                ]
            }
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": "✅ Docker 내부에서 모델 비교 추론 완료 및 .npz 추출"}}]
            }
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": "✅ Windows 로컬 pyMotorenv_310 환경에 시각화 필수 패키지 구성"}}]
            }
        },
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": "🔄 paper_visualizations.ipynb + pyMCAD4SolverX.ipynb 통합 노트북 작업 진행 중"}}]
            }
        }
    ]
}

req = urllib.request.Request(
    f"https://api.notion.com/v1/blocks/{PAGE_ID}/children",
    data=json.dumps(data).encode('utf-8'),
    headers=headers,
    method='PATCH'
)

try:
    with urllib.request.urlopen(req) as response:
        print("Notion Page Updated Successfully!")
except Exception as e:
    print(f"Error updating Notion: {e}")
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))