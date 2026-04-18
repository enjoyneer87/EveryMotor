from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(r"D:\KDH\NvidiaNemo\optiSLang_analysis")
OUT_GIF = OUT_DIR / "MotorCAD_optiSLang_flow_lane_ko.gif"

W, H = 1920, 1080
BG = (245, 247, 250)
BOX_SETUP = (216, 233, 250)
BOX_CI = (214, 242, 232)
BOX_SOLVER = (252, 233, 204)
BOX_ACTIVE = (255, 224, 168)
BOX_DONE = (196, 233, 210)
LINE = (45, 83, 138)
TEXT = (20, 24, 32)

LANE_NAMES = {
    "A": "Lane A: 설정/프로젝트 구성",
    "B": "Lane B: CI 플러그인 런타임",
    "C": "Lane C: Motor-CAD 솔버 실행",
}

STEPS = [
    ("A", "1) Summary 파일 파싱\n(MotorCAD_Summary.txt)"),
    ("A", "2) optiSLang 프로젝트 +\nMotorCAD CI 노드 생성"),
    ("B", "3) 입력 추출\n(scan_input: i_* 탐색)"),
    ("B", "4) 설계별 파라미터 반영\n(set_parameters)"),
    ("B", "5) 솔버 서브프로세스 실행\n(RunOptimisation(0) 호출)"),
    ("C", "6) 헤더 로직 실행\n(Motor-CAD 연결/입력 적용)"),
    ("C", "7) 형상/권선 유효성 검사\n(CheckGeometry/CheckWinding)"),
    ("C", "8) 계산 수행\n(Lab/Mechanical/Duty Cycle)"),
    ("C", "9) 풋터 처리\n(MotorCAD_Outputs.txt 기록)"),
    ("B", "10) 출력 스캔 및 응답 반환\n(scan_output -> optiSLang)"),
]

FLOW = [
    ("A", 0),
    ("A", 1),
    ("B", 0),
    ("B", 1),
    ("B", 2),
    ("C", 0),
    ("C", 1),
    ("C", 2),
    ("C", 3),
    ("B", 3),
]

LANE_X = {
    "A": (60, 620),
    "B": (650, 1270),
    "C": (1300, 1860),
}

LANE_Y_START = 170
BOX_H = 110
BOX_GAP = 24
LANE_BOX_COUNTS = {"A": 2, "B": 4, "C": 4}


def build_lane_rects() -> dict[tuple[str, int], tuple[int, int, int, int]]:
    rects: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    for lane, count in LANE_BOX_COUNTS.items():
        x0, x1 = LANE_X[lane]
        for idx in range(count):
            y0 = LANE_Y_START + idx * (BOX_H + BOX_GAP)
            y1 = y0 + BOX_H
            rects[(lane, idx)] = (x0, y0, x1, y1)
    return rects


RECTS = build_lane_rects()

EDGE_LIST = [
    (("A", 0), ("A", 1)),
    (("A", 1), ("B", 0)),
    (("B", 0), ("B", 1)),
    (("B", 1), ("B", 2)),
    (("B", 2), ("C", 0)),
    (("C", 0), ("C", 1)),
    (("C", 1), ("C", 2)),
    (("C", 2), ("C", 3)),
    (("C", 3), ("B", 3)),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("malgun.ttf", "segoeui.ttf", "arial.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_arrow(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line([p1, p2], fill=color, width=4)
    x2, y2 = p2
    draw.polygon([(x2, y2), (x2 - 12, y2 - 8), (x2 - 12, y2 + 8)], fill=color)


def lane_fill(lane: str) -> tuple[int, int, int]:
    if lane == "A":
        return BOX_SETUP
    if lane == "B":
        return BOX_CI
    return BOX_SOLVER


def step_to_lane_slot(step_index: int) -> tuple[str, int]:
    lane = STEPS[step_index][0]
    slot = sum(1 for i in range(step_index + 1) if STEPS[i][0] == lane) - 1
    return lane, slot


def edge_points(src: tuple[str, int], dst: tuple[str, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    s = RECTS[src]
    d = RECTS[dst]

    if src[0] == dst[0]:
        p1 = ((s[0] + s[2]) // 2, s[3])
        p2 = ((d[0] + d[2]) // 2, d[1])
    elif src[0] < dst[0]:
        p1 = (s[2], (s[1] + s[3]) // 2)
        p2 = (d[0], (d[1] + d[3]) // 2)
    else:
        p1 = (s[0], (s[1] + s[3]) // 2)
        p2 = (d[2], (d[1] + d[3]) // 2)
    return p1, p2


def render_frame(active_idx: int, pulse: float, caption_text: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    title_font = load_font(46)
    lane_font = load_font(26)
    step_font = load_font(26)
    caption_font = load_font(30)

    draw.text((40, 28), "Motor-CAD optiSLang 동작 흐름 (Lane 분리형)", fill=TEXT, font=title_font)
    draw.text((40, 88), "설정 단계 -> CI 실행 -> 솔버 계산 -> 출력 회수 과정을 단계별 애니메이션으로 표현", fill=(70, 80, 100), font=lane_font)

    for lane in ("A", "B", "C"):
        x0, _ = LANE_X[lane]
        draw.text((x0 + 8, 132), LANE_NAMES[lane], fill=(52, 64, 88), font=lane_font)

    for edge_idx, (src, dst) in enumerate(EDGE_LIST):
        p1, p2 = edge_points(src, dst)
        color = (160, 174, 194)
        if active_idx > 0 and edge_idx < active_idx:
            color = (83, 150, 119)
        if active_idx > 0 and edge_idx == active_idx - 1:
            glow = int(25 * pulse)
            color = (45, min(190, 120 + glow), 138)
        draw_arrow(draw, p1, p2, color)

    for idx, (lane, text) in enumerate(STEPS):
        lane_key, lane_slot = step_to_lane_slot(idx)
        rect = RECTS[(lane_key, lane_slot)]

        if idx < active_idx:
            fill = BOX_DONE
        elif idx == active_idx:
            gain = int(20 * pulse)
            fill = (min(255, BOX_ACTIVE[0] + gain), BOX_ACTIVE[1], BOX_ACTIVE[2])
        else:
            fill = lane_fill(lane)

        draw.rounded_rectangle(rect, radius=16, fill=fill, outline=LINE, width=3)
        draw.multiline_text((rect[0] + 16, rect[1] + 16), text, fill=TEXT, font=step_font, spacing=6)

    bar_x0, bar_y0, bar_x1, bar_y1 = 80, 920, 1840, 960
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=10, fill=(225, 229, 236), outline=(180, 188, 202), width=2)
    progress = max(0.0, (active_idx + 1) / len(STEPS))
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * progress)
    draw.rounded_rectangle((bar_x0, bar_y0, fill_x, bar_y1), radius=10, fill=(94, 142, 199), outline=None)

    draw.text((80, 980), caption_text, fill=TEXT, font=caption_font)
    return img


def build_gif() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    durations = []

    for _ in range(8):
        frames.append(render_frame(-1, 0.0, "동작 흐름 설명을 시작합니다..."))
        durations.append(140)

    for idx in range(len(STEPS)):
        for k in range(8):
            pulse = abs(0.5 - (k / 7.0)) * 2.0
            caption = f"단계 {idx + 1}/{len(STEPS)}: {STEPS[idx][1].replace(chr(10), ' / ')}"
            frames.append(render_frame(idx, pulse, caption))
            durations.append(120)

    for _ in range(12):
        frames.append(render_frame(len(STEPS) - 1, 0.0, "완료: 계산 결과가 optiSLang 응답 값으로 전달되었습니다."))
        durations.append(120)

    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


if __name__ == "__main__":
    build_gif()
    print(f"Created: {OUT_GIF}")
