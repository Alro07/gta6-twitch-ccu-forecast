"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (hours 2..24).

Model (built from the case's own comparables, no network, stdlib only):

  * Work in log space.  For every comparable past launch, take its release-day
    hourly curve and form the log-ratio log(viewers_h / viewers_1) for each hour
    h, anchored on that launch's own hour-1 print.  A launch whose whole
    release day is not in the file, or whose name is this case's own game (the
    GTA VI pre-release history is not a launch day), is left out.
  * For each hour, fit a small weighted ridge regression of the log-ratio on
    three features: an intercept, log10(hour-1 viewers) centred at 100k, and a
    console-only flag.  The size feature matters: across the comparables, the
    bigger hour 1 was, the flatter the rest of the day ran (a launch that
    already drew everybody at midnight has less headroom left to grow), while
    console-only launches run a higher evening peak.  Each launch is weighted
    by how close its hour-1 scale is to this case's (a Gaussian kernel on
    log10 viewers, floored so every launch still speaks), boosted for
    console-only launches and for rolling midnight-local unlocks, and
    discounted when the release was early-access contaminated.
  * The fitted curve is shrunk 15% toward the weighted median curve (guards
    against a noisy regression when few comparables speak) and lightly
    smoothed across neighbouring hours.
  * The spread for an hour comes from the weighted spread of the regression
    residuals at that hour, scaled by 0.75 and floored at 0.20 + 0.03*sqrt(h-1)
    log units, so the band widens over the day.  Quantiles are
    median * exp(z_tau * s_h) with the standard-normal z's.
  * The case's own history is used only as a guard: hour 1's print is floored
    at 1, and when the history shows the category was already drawing a large
    audience before midnight, a degenerate hour-1 print is not trusted blindly
    (see _guard_p1).  The pre-release trend itself was tested on the
    comparables and does not predict the day's shape, so it is not allowed to
    move the forecast.

I/O contract (see the kit's starter for the shape): read every *.in in
$MAKEFUN_INPUT_DIR (else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output) as {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}
for hours 2..24.  Deterministic: no randomness anywhere, fixed iteration
order, so two runs write identical bytes.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Regression / blending knobs, tuned by leave-one-out scoring over the
# comparables in the practice cases (see README).
RIDGE = 1.0          # ridge penalty on the non-intercept regression terms
SPREAD_MULT = 0.75   # scales the residual-based spread
BLEND = 0.85         # weight on the regression curve vs the median curve
SMOOTH = 0.4         # hour-to-hour smoothing of the log-ratio curve
SIZE_BANDWIDTH = 1.0  # dex, Gaussian kernel on log10(hour-1 viewers)
SIZE_FLOOR = 0.02    # every comparable keeps at least this weight


def _parse_time(text):
    """ISO-8601 timestamp (with a trailing Z) to a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _release_day(comp):
    """The comparable's release-day hourly viewer counts, hours 1..24, or None.

    Hours are taken by their offset from the comparable's day_start; a launch
    whose day is incomplete (the GTA VI pre-release history, whose day_start
    is a premiere date, not a release) is skipped.
    """
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            when = _parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(viewers, (int, float)) or viewers is None:
            continue
        offset = (when - day_start).total_seconds() / 3600.0
        if offset == int(offset) and 0 <= offset < 24:
            by_hour[int(offset)] = max(1.0, float(viewers))
    if len(by_hour) < 24:
        return None
    return [by_hour[h] for h in range(24)]


def _weight(comp, p1_case):
    """How much one comparable launch counts for this case."""
    log_gap = math.log10(max(1.0, comp["_p1"])) - math.log10(max(1.0, p1_case))
    w = math.exp(-0.5 * (log_gap / SIZE_BANDWIDTH) ** 2)
    w = max(w, SIZE_FLOOR)
    platforms = (comp.get("platforms") or "").upper()
    unlock = (comp.get("unlock") or "").lower()
    if "PC" not in platforms:
        w *= 1.5  # console-only launches, like GTA VI (PS5 / Xbox Series)
    if "midnight local" in unlock:
        w *= 1.2  # rolling midnight unlocks match GTA VI's pattern
    if "early access" in unlock:
        w *= 0.6  # early-access days are not clean release days
    return w


def _weighted_median(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return statistics.median([v for v, _ in pairs])
    acc = 0.0
    for value, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return value
    return pairs[-1][0]


def _solve3(matrix, rhs):
    """Solve a 3x3 linear system by Gaussian elimination; None if singular."""
    rows = [list(matrix[i]) + [rhs[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-12:
            return None
        for r in range(col + 1, 3):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, 4):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * 3
    for i in range(2, -1, -1):
        out[i] = (rows[i][3] - sum(rows[i][j] * out[j] for j in range(i + 1, 3))) / rows[i][i]
    return out


def _features(p1, console_only):
    return [1.0, math.log10(max(1.0, p1)) - 5.0, 1.0 if console_only else 0.0]


def _guard_p1(p1, history_hourly):
    """Sanity-guard hour 1's print against the pre-release history.

    A print below 1 counts as 1.  If the print is tiny while the category was
    already drawing thousands of viewers an hour in the hours before
    midnight, the print is almost certainly a partial read, so the
    pre-release level is used instead of a near-zero anchor.
    """
    p1 = max(1.0, float(p1))
    if not history_hourly:
        return p1
    pre = []
    for row in history_hourly:
        viewers = row.get("viewers")
        if isinstance(viewers, (int, float)) and viewers and viewers > 0:
            pre.append(float(viewers))
    if len(pre) >= 3 and p1 < 100.0:
        level = statistics.median(pre[-6:])
        if level > 10.0 * p1:
            return max(p1, level)
    return p1


def forecast_case(case, name):
    """Build the 23-hour forecast for one case dict."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = _guard_p1(first["viewers"], case.get("history_hourly"))

    game_name = (case.get("game") or {}).get("name") or ""
    comps = []
    for comp in case.get("comparables") or []:
        if game_name and game_name in (comp.get("name") or ""):
            continue  # this case's own game is not a past launch
        day = _release_day(comp)
        if day is None:
            continue
        entry = dict(comp)
        entry["_day"] = day
        entry["_p1"] = max(1.0, day[0])
        entry["_w"] = _weight(entry, p1)
        comps.append(entry)
    if not comps:
        raise ValueError(f"{name} has no usable comparables")

    # The size feature is clamped to the range the comparables actually span,
    # so an hour-1 print beyond every past launch (GTA VI may well be one)
    # flattens the curve no further than the biggest launch on record.
    lo = min(math.log10(c["_p1"]) for c in comps)
    hi = max(math.log10(c["_p1"]) for c in comps)
    case_log = min(max(math.log10(p1), lo), hi)
    # GTA VI is console-only (PS5 / Xbox Series X|S, no PC at launch).
    case_x = [1.0, case_log - 5.0, 1.0]

    med_log = []
    spread = []
    for h in range(2, 25):
        rows = []
        ys = []
        ws = []
        for c in comps:
            console_only = "PC" not in (c.get("platforms") or "").upper()
            rows.append(_features(c["_p1"], console_only))
            ys.append(math.log(c["_day"][h - 1] / c["_p1"]))
            ws.append(c["_w"])
        # Weighted ridge regression of this hour's log-ratio on the features.
        a = [[0.0] * 3 for _ in range(3)]
        b = [0.0] * 3
        for x, y, w in zip(rows, ys, ws):
            for i in range(3):
                b[i] += w * x[i] * y
                for j in range(3):
                    a[i][j] += w * x[i] * x[j]
        for i in range(1, 3):
            a[i][i] += RIDGE
        beta = _solve3(a, b)
        med = _weighted_median(list(zip(ys, ws)))
        if beta is None:
            pred = med
            resid = [y - med for y in ys]
        else:
            pred = sum(beta[i] * case_x[i] for i in range(3))
            resid = [y - sum(beta[i] * x[i] for i in range(3)) for x, y in zip(rows, ys)]
        wsum = sum(ws)
        mean_resid = sum(w * r for w, r in zip(ws, resid)) / wsum
        var = sum(w * (r - mean_resid) ** 2 for w, r in zip(ws, resid)) / wsum
        spread.append(max(0.20, math.sqrt(max(var, 0.0))))
        med_log.append(BLEND * pred + (1.0 - BLEND) * med)

    # Light smoothing across neighbouring hours; endpoints stay as fitted.
    smoothed = list(med_log)
    for i in range(1, len(med_log) - 1):
        smoothed[i] = (1.0 - SMOOTH) * med_log[i] + SMOOTH * 0.5 * (med_log[i - 1] + med_log[i + 1])

    hours = []
    for i, h in enumerate(range(2, 25)):
        median = p1 * math.exp(smoothed[i])
        s = max(spread[i] * SPREAD_MULT, 0.20 + 0.03 * math.sqrt(h - 1))
        qs = []
        for z in Z:
            q = median * math.exp(z * s)
            qs.append(min(max(q, 1.0), 1e9))
        qs.sort()
        hours.append({"hour": h, "q": [round(q, 2) for q in qs]})
    return {"hours": hours}


def _fallback_forecast(case):
    """House-style persistence forecast, used only if the model fails.

    Never let one bad case sink the entry: an invalid case scores as every
    quantile equal to hour 1's print and takes the entry out of the prizes,
    so a crash must degrade to a valid forecast, never to no output.
    """
    p1 = 1.0
    try:
        first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
        if first is not None:
            p1 = max(1.0, float(first["viewers"]))
    except (TypeError, ValueError):
        p1 = 1.0
    hours = []
    for h in range(2, 25):
        s = 0.15 * math.sqrt(h - 1)
        qs = sorted(min(max(p1 * math.exp(z * s), 1.0), 1e9) for z in Z)
        hours.append({"hour": h, "q": [round(q, 2) for q in qs]})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = _fallback_forecast(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
