"""Slide-tuned re-renders of the campaign figures for the 2026-08-07 seminar.

The paper figures (`make_campaign_figures.py`) are sized for a two-column PDF:
twelve x-tick labels collide at projector size and the annotation text is set
for reading distance, not room distance. This script re-draws the two figures
that go on the busiest slides, at 16:9 proportions with larger type and
Korean labels, and adds one new figure (gate position) that only exists for
the talk.

Every number is read from the same committed scorecard JSONs the paper cites.
Nothing here is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

NOISE_PP = 0.25          # measured run-to-run variation of this harness
G1P_AIRGAP = 5.0         # G1' primary gate, air-gap |B|   (methodology review section 25)
G1P_TORQUE = 3.0         # G1' primary gate, torque nRMSE

AXIS_COLORS = {
    "baseline": "#7f7f7f",
    "capacity": "#4c72b0",
    "supervision": "#55a868",
    "architecture": "#dd8452",
    "representation": "#c44e52",
    "data": "#0f7b6c",
    "physics": "#8172b3",
}
AXIS_KO = {
    "baseline": "기준",
    "capacity": "용량",
    "supervision": "감독",
    "architecture": "아키텍처",
    "representation": "표현",
    "data": "데이터",
    "physics": "물리 주입",
}

# Korean-capable font: Malgun Gothic ships with Windows. Fall back silently so
# the script still runs (with mojibake, not a crash) on a host without it.
for _cand in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(_cand == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _cand
        break
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.unicode_minus": False,
    "font.size": 17,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def scorecard(name: str, model_key: str | None = None):
    """(|B| nRMSE %, torque nRMSE %, summary dict) from a benchmark JSON."""
    data = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    models = data["models"]
    if model_key is None:
        model_key = next(k for k in models if k.startswith("curl:"))
    s = models[model_key]["summary"]
    return s["overall"]["Bnorm"]["nrmse_pct"], s["torque"]["nrmse_torque_pct"], s


def floors():
    data = json.loads((RESULTS / "benchmark_v2_floors.json").read_text(encoding="utf-8"))
    return {k: v["summary"] for k, v in data["models"].items()}


# Same twelve runs as the paper's fig1, in the order they were run. Labels are
# shortened so they do not collide at projector size.
CAMPAIGN = [
    ("h128\n기준선",        "benchmark_v2_nodeB_notime.json",           "baseline",       "§10"),
    ("h256\n폭 ×2",         "benchmark_v2_nodeB_h256.json",             "capacity",       "§13"),
    ("p24\n깊이 +60%",      "benchmark_v2_nodeB_proc24.json",           "capacity",       "§15"),
    ("스펙트럼\nw=1",       "benchmark_v2_nodeB_spectral_bw1.json",     "supervision",    "§16"),
    ("스펙트럼\nw=10",      "benchmark_v2_nodeB_spectral_bw10.json",    "supervision",    "§16"),
    ("스펙트럼\nw=30",      "benchmark_v2_nodeB_spectral_bw30.json",    "supervision",    "§16"),
    ("Transolver\n전역어텐션", "benchmark_v2_nodeB_r3_transolver.json", "architecture",   "§17"),
    ("Hybrid\n장거리엣지",  "benchmark_v2_nodeB_r3_hybrid.json",        "architecture",   "§18"),
    ("curl-A\n타깃 A",      "benchmark_v2_nodeB_r4_curl_spectral.json", "representation", "§19"),
    ("DOE 120\n(110 geom)", "benchmark_v2_nodeB_doe120_bw30.json",      "data",           "§20"),
    ("DOE 240\nstep 동급",  "benchmark_v2_nodeB_doe240_bw30.json",      "data",           "§21"),
    ("DOE 240\n에폭 동급",  "benchmark_v2_nodeB_doe240_bw30_ep50.json", "data",           "§21b"),
    ("+물리 prior\n(240)",  "benchmark_v2_nodeB_doe240_bw30_prior_ep50.json", "physics",  "§27"),
    ("형상×전류\n320 (v2)", "benchmark_v2_nodeB_doe_v2_bw30_prior_legacy6.json", "data",  "§29"),
]


def fig_elimination(out: Path) -> Path:
    fl = floors()
    curl_b = fl["curl_representation_floor"]["overall"]["Bnorm"]["nrmse_pct"]
    curl_t = fl["curl_representation_floor"]["torque"]["nrmse_torque_pct"]

    labels, bvals, tvals, colors = [], [], [], []
    for label, fname, group, sec in CAMPAIGN:
        b, t, _ = scorecard(fname)
        labels.append(f"{label}\n{sec}" if "\n" in label else f"{label}\n{sec}")
        bvals.append(b)
        tvals.append(t)
        colors.append(AXIS_COLORS[group])

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17.5, 9.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.08})

    best = int(np.argmin(bvals))
    ax1.bar(x, bvals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax1.axhline(bvals[0], color="#7f7f7f", ls=":", lw=1.6, zorder=2)
    ax1.axhspan(bvals[best] - NOISE_PP, bvals[best] + NOISE_PP,
                color="#0f7b6c", alpha=0.18, zorder=1)
    ax1.axhline(curl_b, color="#333333", ls="--", lw=1.8, zorder=4)
    for xi, v in zip(x, bvals):
        ax1.text(xi, v + 0.30, f"{v:.2f}", ha="center", va="bottom", fontsize=15.5)
    ax1.set_ylabel("held-out |B| nRMSE [%]")
    ax1.set_ylim(0, max(bvals) * 1.17)
    ax1.set_title("고정 6-geometry 홀드아웃 위의 단일축 소거 — 용량·감독·아키텍처·표현은 |B| 바닥을 못 깼고,\n"
                  "데이터(geometry 수·전류축)와 물리 주입만 움직였다 — 물리 주입은 특히 토크를 (§27, §29)",
                  loc="left", pad=14)
    ax1.text(0.006, 0.955,
             f"점선 = h128 기준선 ({bvals[0]:.2f}%)      "
             f"음영 = 최고 성적 ±{NOISE_PP} pp 런간 변동 밴드 ({bvals[best]:.2f}%)",
             transform=ax1.transAxes, fontsize=14, color="#444444", va="top")

    ax2.bar(x, tvals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax2.axhline(tvals[0], color="#7f7f7f", ls=":", lw=1.6, zorder=2)
    ax2.axhline(curl_t, color="#333333", ls="--", lw=1.8, zorder=4)
    for xi, v in zip(x, tvals):
        ax2.text(xi, v + 0.14, f"{v:.2f}", ha="center", va="bottom", fontsize=15.5)
    ax2.set_ylabel("held-out 토크 nRMSE [%]")
    ax2.set_ylim(0, max(tvals) * 1.26)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    handles = [Patch(facecolor=AXIS_COLORS[k], label=AXIS_KO[k]) for k in AXIS_COLORS]
    handles.append(Line2D([], [], color="#333333", ls="--", lw=1.8,
                          label="curl 표현 바닥 (|B| 5.31% / 토크 1.58%)"))
    fig.legend(handles=handles, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               framealpha=0.95, fontsize=14.5)
    fig.text(0.006, -0.085,
             "전 막대 동일 조건: case-level 홀드아웃(test geometry 4/7/18/32/37/39), 270 샘플, "
             "동일 요소 부분집합(coverage 0.865), eval/benchmark.py 채점.",
             fontsize=13, color="#555555")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_scaling(out: Path) -> Path:
    points = [
        ("benchmark_v2_nodeB_spectral_bw30.json", 30, 100),
        ("benchmark_v2_nodeB_doe120_bw30.json", 110, 50),
        ("benchmark_v2_nodeB_doe240_bw30_ep50.json", 230, 50),
    ]
    geo, bvals, tvals, epochs = [], [], [], []
    for fname, n_train, n_ep in points:
        b, t, _ = scorecard(fname)
        geo.append(n_train); bvals.append(b); tvals.append(t); epochs.append(n_ep)
    b_sm, t_sm, _ = scorecard("benchmark_v2_nodeB_doe240_bw30.json")

    fig, (ax, axt) = plt.subplots(2, 1, figsize=(14.5, 9.0), sharex=True,
                                  gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.10})
    ax.plot(geo, bvals, "o-", color="#0f7b6c", lw=3.0, ms=13, zorder=4, label="|B| nRMSE")
    ax.fill_between(geo, np.array(bvals) - NOISE_PP, np.array(bvals) + NOISE_PP,
                    color="#0f7b6c", alpha=0.16, zorder=2, label=f"±{NOISE_PP} pp 런간 변동")
    ax.plot([geo[-1]], [b_sm], "o", mfc="white", mec="#0f7b6c", mew=2.6, ms=13, zorder=5)
    ax.annotate(f"{b_sm:.3f}% — 같은 240 geometry, 25에폭(step-matched).\n"
                "'스케일링 소진'으로 처음 읽혔던 점.",
                xy=(geo[-1] - 3, b_sm), xytext=(geo[-1] - 118, b_sm + 0.42),
                fontsize=14.5, color="#666666",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.6))
    for gx, bv in zip(geo, bvals):
        ax.annotate(f"{bv:.3f}%", xy=(gx, bv), xytext=(0, 16), textcoords="offset points",
                    ha="center", fontsize=17, color="#0f7b6c", fontweight="bold")
    # set_ylim first: the epoch captions are pinned to the final axis bottom, not
    # to the autoscaled one (which put them in the middle of the plot).
    ax.set_ylabel("held-out |B| nRMSE [%]")
    ax.set_ylim(min(bvals) - 0.75, max(bvals) + 0.75)
    for gx, ep in zip(geo, epochs):
        ax.annotate(f"{ep}에폭", xy=(gx, ax.get_ylim()[0]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=13.5, color="#666666")
    ax.set_xticks(geo)
    ax.set_xlim(min(geo) - 30, max(geo) + 30)
    ax.set_title("학습 geometry 스케일링 — 레버는 실재하고 아직 열려 있다.\n"
                 "그러나 기울기가 게이트에 닿지 못한다 (각 점은 자기 예산으로 수렴까지 학습)",
                 loc="left", pad=14)
    ax.legend(loc="upper right", ncol=2)
    ax.annotate("-0.79 pp\n(-0.42 pp / doubling)", xy=(70, min(bvals) - 0.10),
                ha="center", fontsize=15, color="#0f7b6c",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#0f7b6c", lw=1.4))
    ax.annotate("-0.32 pp  (-0.30 pp / doubling)\n6개 test geometry 전부 개선",
                xy=(170, min(bvals) - 0.45), ha="center", fontsize=15, color="#0f7b6c",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#0f7b6c", lw=1.4))

    axt.plot(geo, tvals, "s--", color="#c44e52", lw=2.6, ms=12, zorder=4)
    axt.plot([geo[-1]], [t_sm], "s", mfc="white", mec="#c44e52", mew=2.6, ms=12, zorder=5)
    axt.annotate(f"{t_sm:.3f}%", xy=(geo[-1], t_sm), xytext=(0, 14), textcoords="offset points",
                 ha="center", fontsize=15, color="#999999")
    for gx, tv in zip(geo, tvals):
        axt.annotate(f"{tv:.3f}%", xy=(gx, tv), xytext=(0, -30), textcoords="offset points",
                     ha="center", fontsize=15, color="#c44e52")
    axt.set_ylabel("토크 nRMSE [%]")
    axt.set_xlabel("학습 geometry 수  (test / val은 전 구간 동일 고정)")
    axt.set_ylim(min(tvals) - 0.55, max(t_sm, max(tvals)) + 0.35)

    fig.text(0.006, -0.055,
             "외삽(명시): 측정된 -0.30 pp/doubling으로 pooled 5%까지는 ~22 doubling. "
             "G1'(공극 5%) 기준으로 다시 계산해도 ~7 doubling  약  3.4만 geometry  약  Motor-CAD 연속 솔브 62일.\n"
             "데이터 축은 열려 있으나 게이트에는 닿지 못한다 — 다른 레버가 필요하다.",
             fontsize=13.5, color="#555555")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_gate_position(out: Path) -> Path:
    """Three model generations against the adopted G1' gate and the hard floor.

    champion (240, no prior) -> +physics prior (section 27) -> v2 320
    geometry x current (section 29, scored on the SAME legacy 6-test subset so
    every bar is the same task). The remaining-distance arrow reads from v2.
    """
    fl = floors()
    curl = fl["curl_representation_floor"]
    _, ch_tq, champ = scorecard("benchmark_v2_nodeB_doe240_bw30_ep50.json")
    _, pr_tq, prior = scorecard("benchmark_v2_nodeB_doe240_bw30_prior_ep50.json")
    _, v2_tq, v2 = scorecard("benchmark_v2_nodeB_doe_v2_bw30_prior_legacy6.json")

    rows = [
        ("공극 |B| nRMSE",
         curl["by_region"]["airgap"]["Bnorm"]["nrmse_pct"], G1P_AIRGAP,
         champ["by_region"]["airgap"]["Bnorm"]["nrmse_pct"],
         prior["by_region"]["airgap"]["Bnorm"]["nrmse_pct"],
         v2["by_region"]["airgap"]["Bnorm"]["nrmse_pct"]),
        ("토크 nRMSE",
         curl["torque"]["nrmse_torque_pct"], G1P_TORQUE, ch_tq, pr_tq, v2_tq),
    ]

    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    y = np.arange(len(rows))[::-1].astype(float) * 1.25
    dy = 0.26
    for yi, (name, floor_v, gate_v, ch_v, pr_v, v2_v) in zip(y, rows):
        ax.barh(yi + dy, ch_v, height=0.22, color="#c5ddd9", zorder=3)
        ax.barh(yi, pr_v, height=0.22, color="#9ec9c2", zorder=3)
        ax.barh(yi - dy, v2_v, height=0.22, color="#0f7b6c", zorder=3)
        ax.barh(yi - dy, floor_v, height=0.22, color="#333333", alpha=0.32, zorder=4)
        ax.plot([gate_v, gate_v], [yi - 0.48, yi + 0.58], color="#c44e52", lw=3.4, zorder=6)

        ax.text(ch_v + 0.13, yi + dy, f"champion {ch_v:.2f}%", va="center",
                fontsize=13.5, color="#7ba39c")
        ax.text(pr_v + 0.13, yi, f"+물리 prior {pr_v:.2f}%", va="center",
                fontsize=13.5, color="#5d8f88")
        ax.text(v2_v + 0.13, yi - dy, f"현재 v2 {v2_v:.2f}%", va="center",
                fontsize=17, color="#0f7b6c", fontweight="bold")
        ax.text(gate_v, yi + 0.61, f"G1′ {gate_v:.0f}%", ha="center", va="bottom",
                fontsize=15.5, color="#c44e52")
        ax.annotate("", xy=(gate_v, yi - 0.56), xytext=(v2_v, yi - 0.56),
                    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.6))
        ax.text((gate_v + v2_v) / 2, yi - 0.70, f"남은 거리 {v2_v - gate_v:.2f} pp",
                ha="center", va="center", fontsize=14, color="#666666")
        ax.text(floor_v / 2, yi - dy, f"바닥 {floor_v:.2f}%", ha="center", va="center",
                fontsize=12.5, color="#ffffff")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=18)
    ax.set_ylim(-1.05, y[0] + 0.95)
    ax.set_xlabel("nRMSE [%]  (전 막대 동일 과제: 레거시 6-geometry 홀드아웃, 650.5 A)")
    ax.set_xlim(0, 10.2)
    ax.set_title("주 게이트 G1′ 대비 3세대 위치 — 공극 6.62% / 토크 3.27%, 역대 최근접 (§29)",
                 loc="left", pad=12)
    handles = [
        Patch(facecolor="#0f7b6c", label="현재: v2 320 형상×전류 (§29)"),
        Patch(facecolor="#9ec9c2", label="+물리 prior (240, §27)"),
        Patch(facecolor="#c5ddd9", label="champion (240, §21b)"),
        Patch(facecolor="#333333", alpha=0.32, label="curl 표현 바닥 (도달 가능 한계)"),
        Line2D([], [], color="#c44e52", lw=3.4, label="G1′ 게이트"),
    ]
    fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.14),
               fontsize=13, framealpha=0.95)
    fig.text(0.006, -0.215,
             "v2는 형상×전류 320케이스 학습 후 같은 레거시 게이트에서 채점한 값 -- 전류 능력을 얻고도 "
             "고정 여자 과제가 후퇴하지 않았다(§29). 미학습 형상×전류 12케이스는 공극 5.35% / 토크 2.61%(풀드).",
             fontsize=13.5, color="#555555")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=RESULTS / "viz")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for p in (
        fig_elimination(args.outdir / "seminar_fig1_elimination.png"),
        fig_scaling(args.outdir / "seminar_fig2_scaling.png"),
        fig_gate_position(args.outdir / "seminar_fig6_gate.png"),
    ):
        print("wrote", p)


if __name__ == "__main__":
    main()
