"""Render saved terminal scalar context only; never run a model or native solver."""
import argparse
import json
from pathlib import Path


def Render(root: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['prior_completed_training']
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    colors = ['#65758b', '#087e8b', '#aa6b28']
    for i, row in enumerate(rows):
        values = row['primary_Bvector']
        ax.hlines(i, values['min'], values['max'], color=colors[i], linewidth=6, alpha=.35)
        ax.scatter(list(values['by_seed'].values()), [i] * 3, marker='o', s=55, color=colors[i])
        ax.scatter(values['mean'], i, marker='D', s=80, color=colors[i], edgecolors='white', zorder=3)
        ax.text(13.2, i, f"Mean {values['mean']:.2f}%", va='center', fontsize=11)
    ax.set_yticks(range(3), [r['model'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 15.5)
    ax.set_ylim(2.6, -.6)
    ax.set_xlabel('Whole B-vector nRMSE (%) — lower is better')
    ax.set_title('Previously completed training | saved final results', loc='left', fontweight='bold', pad=16)
    ax.grid(axis='x', alpha=.2)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    fig.text(.03, .075, 'Dots: fixed seeds 17 / 29 / 43. Diamond: mean. Line: min–max range, NOT a confidence interval.', fontsize=9)
    fig.text(.03, .035, 'Exposed innerdev32, 45 stored samples. R3 new training: 0. No native boundary qualification.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    (root / 'assets').mkdir(exist_ok=True)
    fig.savefig(root / 'assets' / 'prior_completed_results.png', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    Render(parser.parse_args().root)
