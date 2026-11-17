"""GTA VI launch-day Twitch viewers forecaster (Python 3.13, standard library only).

Model: a launch-day shape learned at runtime from the case file's own
comparables (past big game launches' hourly Twitch curves).

For each comparable launch we take its release-day curve (UTC hours 1..24)
and express every hour as a log-ratio to that launch's hour-1 print. Across
launches, the log-ratio of hour h is regressed on the launch's size,
ln(hour-1 print): big launches unfold flatter (the peak-to-hour-1 ratio
falls as hour 1 grows, correlation about -0.75 on the comparable set), so
the regression's slope grows from ~0 in the early hours to about -0.4 at
the evening peak. The forecast for the case is then:

    median log-ratio(h) = a_h + b_h * (ln(p1) - mean ln p1)
    q_tau(h) = p1 * exp(median + z_tau * s_h)

where p1 is the case's hour-1 print, z_tau the standard normal quantiles
(-1.645, -0.674, 0, 0.674, 1.645), and s_h a robust per-hour spread (the
interquartile range of the regression residuals / 1.349). The centre uses
the median residual, because the cross-launch distribution of log-ratios
is right-skewed (a few launches run far above the rest). ln(p1) is clamped
to the training launches' range, so a record-size hour 1 never extrapolates
the slope past the biggest launch on record.

Which comparables are used (decided from the case file itself, never from
names): the launch's unlock note must not mark it early-access-contaminated,
weak, or not-a-launch, and its hour-1 print must be at least 5,000 viewers
(a near-dead category at 00:00 UTC means the launch unlocked later in the
day, a different regime). Early-access launches are excluded because their
hour 1 is already inflated; simultaneous-global unlocks are excluded by the
hour-1 floor (their hour 1 is a near-empty category). If too few
comparables survive for an hour, that hour falls back to the house
persistence shape (spread 0.15 * sqrt(h-1) in log space).

Practice (four past launches, scored with the same pinball loss on the log
scale as the final): mean loss 0.171 vs the house baseline's 0.387;
leave-one-launch-out over the comparable regime: 0.242. Each hour's
centre, slope and spread are lightly smoothed across neighbouring hours
(the launch-day shape is smooth in h; per-hour fits are noisy), and every
output row goes through one cleaning step that enforces the contest's
bounds, so no path — fitted or fallback — can emit an out-of-range
quantile. The model uses only the case file: no network, no wall clock,
no randomness, so two runs produce byte-identical output.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim,
standard library only. It reads every <id>.in in the input directory and
writes <id>.out to the output directory: {"hours": [{"hour": h,
"q": [q05, q25, q50, q75, q95]}, ...]} for hours 2..24.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an
# `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# A comparable counts only when its hour-1 print is at least this many
# viewers: below it the category was near-dead at midnight UTC, which
# means the launch unlocked later in the day (a different shape regime).
MIN_HOUR1 = 5000
# Minimum usable launches for a per-hour fit; below it the hour falls back
# to the house persistence shape.
MIN_FIT = 8
# The house baseline's log spread per square-root hour, used only as a
# fallback when an hour has too few usable comparables.
BASELINE_SIGMA = 0.15
# Smallest spread allowed, in log units: the bands never collapse to a point.
MIN_SPREAD = 0.05


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp (with a trailing Z) as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comparable: dict) -> dict:
    """A comparable's release-day hourly viewers as {hour: viewers}.

    Hours run 1..24 from the launch's day_start (00:00 UTC of its release
    date). Points the source did not sample are simply absent.
    """
    day_start = parse_time(comparable["day_start"])
    curve = {}
    for row in comparable.get("hourly") or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        at = parse_time(row["from"])
        delta = (at - day_start).total_seconds()
        hour = int(delta // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = viewers
    return curve


def usable(comparable: dict) -> bool:
    """Whether a comparable belongs to the launch regime being forecast.

    Excluded: launches whose unlock note marks them early-access
    contaminated, weak analogues, or not a launch at all (the target
    game's own pre-release category history). The hour-1 floor is checked
    by the caller, from the curve.
    """
    note = (comparable.get("unlock") or "").lower()
    if "early-access-contaminated" in note:
        return False
    if "tier: weak" in note:
        return False
    if "not a launch" in note:
        return False
    return True


def fit_hour(pairs: list) -> tuple:
    """Fit one hour's shape from (ln p1, log-ratio) pairs.

    Returns (centre, slope, spread, mean ln p1, min ln p1, max ln p1).
    The centre is the regression line evaluated at the median residual, so
    it estimates the conditional median of the log-ratio; the spread is
    the residuals' interquartile range scaled to a normal standard
    deviation, which resists the few launches that run far above the rest.
    """
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if denom > 0:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, ys)]
    centre = mean_y + statistics.median(residuals)
    if len(residuals) >= 4:
        quartiles = statistics.quantiles(residuals, n=4)
        spread = (quartiles[2] - quartiles[0]) / 1.349
    else:
        spread = statistics.pstdev(residuals) if len(residuals) > 1 else MIN_SPREAD
    spread = max(spread, MIN_SPREAD)
    return (centre, slope, spread, mean_x, min(xs), max(xs))


def clean_quantiles(values: list) -> list:
    """Raw quantile values as one valid output row: capped at the
    contest's 1e9 ceiling, floored at 0.01, rounded like the grader's
    JavaScript, and forced non-decreasing. Both the fitted branch and
    the fallback branch go through this, so no path can emit an
    out-of-range quantile: the fallback used to cap nothing, and a huge
    hour-1 print with no usable comparables produced q95 above 1e9
    (an entry with any invalid case takes no prize)."""
    out = []
    for value in values:
        value = min(value, 1_000_000_000)
        value = max(value, 0.01)
        out.append(round2(value))
    # Rounding and capping keep the row non-decreasing (the unrounded
    # values already are); enforce it outright so the file is always valid.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def hour_quantiles(p1: float, hour: int, fit: tuple | None) -> list:
    """The five quantiles for one hour: the fitted shape, or the house
    persistence shape when the hour had too few usable comparables."""
    if fit is None:
        return clean_quantiles(
            [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(hour - 1)) for z in Z]
        )
    centre, slope, spread, mean_x, low_x, high_x = fit
    ln_p1 = math.log(p1)
    # Never extrapolate the size slope past the biggest or smallest
    # launch the fit was trained on.
    ln_p1 = max(low_x, min(high_x, ln_p1))
    median = centre + slope * (ln_p1 - mean_x)
    return clean_quantiles(
        [p1 * math.exp(median + z * spread) for z in Z]
    )


def smooth_fits(fits: dict) -> dict:
    """Lightly smooth each fitted hour's centre, slope and spread
    across its neighbouring hours.

    The true launch-day shape is smooth in h, but every hour is fitted
    independently from ~19 launches, so each hour's parameters carry
    independent estimation noise. A [0.25, 0.5, 0.25] kernel over h-1,
    h, h+1 pools that noise out without blurring the shape (the fitted
    centres rise smoothly already). Hours without a fit stay without
    one, and a fitted hour whose neighbours have no fit is left alone.
    Measured on the calibration set: LOOCV 0.242 (was 0.243), practice
    0.171 (was 0.171) — no harm on either criterion, and less
    hour-to-hour jitter in the forecast."""
    kernel = (0.25, 0.5, 0.25)
    offsets = (-1, 0, 1)
    smoothed = {}
    for hour, fit in fits.items():
        if fit is None:
            smoothed[hour] = None
            continue
        centre, slope, spread, mean_x, low_x, high_x = fit
        params = []
        for index, value in enumerate((centre, slope, spread)):
            numerator = 0.0
            denominator = 0.0
            for offset, weight in zip(offsets, kernel):
                neighbour = fits.get(hour + offset)
                if neighbour is not None:
                    numerator += weight * neighbour[index]
                    denominator += weight
            params.append(numerator / denominator if denominator else value)
        smoothed[hour] = (
            params[0], params[1], params[2], mean_x, low_x, high_x,
        )
    return smoothed


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1, first["viewers"]) if first is not None else 1

    # The training set: this case file's own comparables, filtered to the
    # launch regime, each as a release-day curve.
    curves = []
    for comparable in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        if not usable(comparable):
            continue
        curve = release_day_curve(comparable)
        hour1 = curve.get(1)
        if hour1 is None or hour1 < MIN_HOUR1:
            continue
        curves.append(curve)

    fits = {}
    for hour in range(2, 25):
        pairs = []
        for curve in curves:
            hour1 = curve.get(1)
            value = curve.get(hour)
            if hour1 is None or value is None or value <= 0:
                continue
            pairs.append((math.log(hour1), math.log(value / hour1)))
        fits[hour] = fit_hour(pairs) if len(pairs) >= MIN_FIT else None
    fits = smooth_fits(fits)

    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], fits.get(row["hour"]))}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
