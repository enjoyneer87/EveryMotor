"""Field error metrics in physical units.

The previous benchmark reported normalized MSE, which depends on whichever
per-run standardization happened to be applied and is therefore not comparable
across models. Everything here is computed on de-normalized tensors and carries
its unit explicitly:

    Bx, By, |B|   tesla [T]
    A             weber per metre [Wb/m]
    Je            ampere per square metre [A/m^2]

nRMSE convention
----------------
``nrmse = rmse / rms(truth)``, reported in percent. RMS-normalization (not
range-normalization) is used because the reference quantity is a zero-mean
alternating field, where peak-to-peak range is sensitive to a single saturated
element. The choice is recorded in the output JSON so later runs cannot silently
switch convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

CHANNEL_UNITS: Mapping[str, str] = {
    "Bx": "T",
    "By": "T",
    "Bnorm": "T",
    "A": "Wb/m",
    "J": "A/m^2",
    "Je": "A/m^2",
}

NRMSE_CONVENTION = "rmse / rms(truth), percent"


@dataclass(frozen=True)
class ChannelMetric:
    """Immutable error summary for one field channel, in physical units."""

    name: str
    unit: str
    n: int
    rms_truth: float
    rmse: float
    nrmse_pct: float
    mae: float
    max_abs_err: float
    r2: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _finite_pair(pred: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop non-finite samples from both arrays together.

    Non-finite entries are dropped rather than zero-filled: zero-filling a NaN
    prediction scores it as a correct zero-field prediction, which flatters the
    model. The surviving count is reported as ``n`` so silent mass-dropping is
    visible.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    if pred.shape != truth.shape:
        raise ValueError(f"pred/truth shape mismatch: {pred.shape} vs {truth.shape}")
    ok = np.isfinite(pred) & np.isfinite(truth)
    return pred[ok], truth[ok]


def channel_metric(
    pred: np.ndarray,
    truth: np.ndarray,
    name: str,
    unit: Optional[str] = None,
) -> ChannelMetric:
    """Compute the physical-unit error summary for one channel."""
    p, t = _finite_pair(pred, truth)
    n = int(p.size)
    if n == 0:
        nan = float("nan")
        return ChannelMetric(name, unit or CHANNEL_UNITS.get(name, ""), 0, nan, nan, nan, nan, nan, nan)

    err = p - t
    rms_truth = float(np.sqrt(np.mean(t**2)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # An all-zero reference channel has no meaningful relative error.
    nrmse_pct = float(100.0 * rmse / rms_truth) if rms_truth > 0.0 else float("nan")

    var = float(np.var(t))
    r2 = float(1.0 - np.mean(err**2) / var) if var > 0.0 else float("nan")

    return ChannelMetric(
        name=name,
        unit=unit or CHANNEL_UNITS.get(name, ""),
        n=n,
        rms_truth=rms_truth,
        rmse=rmse,
        nrmse_pct=nrmse_pct,
        mae=float(np.mean(np.abs(err))),
        max_abs_err=float(np.max(np.abs(err))),
        r2=r2,
    )


def field_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    channel_names: Sequence[str],
) -> Dict[str, ChannelMetric]:
    """Per-channel metrics for a ``(n_points, n_channels)`` field pair.

    When the channels contain both ``Bx`` and ``By``, a derived ``Bnorm`` channel
    is added — the flux-density magnitude is what saturation and iron-loss
    correlations key off, so it is reported alongside the components.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError(f"pred/truth must be [N, C], got {pred.shape} and {truth.shape}")
    if pred.shape != truth.shape:
        raise ValueError(f"pred/truth shape mismatch: {pred.shape} vs {truth.shape}")
    if len(channel_names) != pred.shape[1]:
        raise ValueError(
            f"{len(channel_names)} channel names for {pred.shape[1]} channels: {list(channel_names)}"
        )

    out: Dict[str, ChannelMetric] = {}
    for i, name in enumerate(channel_names):
        out[name] = channel_metric(pred[:, i], truth[:, i], name)

    names = list(channel_names)
    if "Bx" in names and "By" in names:
        ix, iy = names.index("Bx"), names.index("By")
        pn = np.hypot(pred[:, ix], pred[:, iy])
        tn = np.hypot(truth[:, ix], truth[:, iy])
        out["Bnorm"] = channel_metric(pn, tn, "Bnorm")
    return out


def region_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    channel_names: Sequence[str],
    group_of_point: Sequence[str],
    groups: Optional[Sequence[str]] = None,
    min_points: int = 1,
) -> Dict[str, Dict[str, ChannelMetric]]:
    """Per-region-group, per-channel metrics.

    Groups with fewer than ``min_points`` samples are omitted rather than
    reported with a meaningless denominator.
    """
    group_of_point = np.asarray(group_of_point, dtype=str)
    pred = np.asarray(pred, dtype=np.float64)
    if group_of_point.size != pred.shape[0]:
        raise ValueError(
            f"group_of_point has {group_of_point.size} entries for {pred.shape[0]} points"
        )
    wanted = list(groups) if groups is not None else sorted(set(group_of_point.tolist()))

    out: Dict[str, Dict[str, ChannelMetric]] = {}
    for g in wanted:
        mask = group_of_point == g
        if int(np.count_nonzero(mask)) < min_points:
            continue
        out[g] = field_metrics(pred[mask], np.asarray(truth, dtype=np.float64)[mask], channel_names)
    return out


def metrics_to_dict(metrics: Mapping[str, ChannelMetric]) -> Dict[str, Dict[str, object]]:
    """Serialize a channel-metric mapping for the results JSON."""
    return {name: m.to_dict() for name, m in metrics.items()}


def aggregate_channel(metrics_per_sample: Sequence[Mapping[str, ChannelMetric]], channel: str) -> Dict[str, float]:
    """Aggregate one channel across samples.

    RMSE is pooled as a sample-count-weighted quadratic mean (so it equals the
    RMSE of the concatenated samples); nRMSE is recomputed from the pooled RMSE
    and pooled truth RMS rather than averaged, because averaging ratios weights
    low-field samples disproportionately.
    """
    n_tot = 0.0
    sq_err = 0.0
    sq_truth = 0.0
    per_sample_nrmse = []
    for m in metrics_per_sample:
        cm = m.get(channel)
        if cm is None or cm.n == 0 or not np.isfinite(cm.rmse):
            continue
        n_tot += cm.n
        sq_err += cm.n * cm.rmse**2
        sq_truth += cm.n * cm.rms_truth**2
        if np.isfinite(cm.nrmse_pct):
            per_sample_nrmse.append(cm.nrmse_pct)

    if n_tot == 0.0:
        nan = float("nan")
        return {"n": 0, "rmse": nan, "nrmse_pct": nan, "nrmse_pct_median": nan, "nrmse_pct_p95": nan}

    rmse = float(np.sqrt(sq_err / n_tot))
    rms_truth = float(np.sqrt(sq_truth / n_tot))
    return {
        "n": int(n_tot),
        "rmse": rmse,
        "nrmse_pct": float(100.0 * rmse / rms_truth) if rms_truth > 0 else float("nan"),
        "nrmse_pct_median": float(np.median(per_sample_nrmse)) if per_sample_nrmse else float("nan"),
        "nrmse_pct_p95": float(np.percentile(per_sample_nrmse, 95)) if per_sample_nrmse else float("nan"),
    }
