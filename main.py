"""GTA VI Twitch CCU forecast v2: comparables 3-feature log-ratio regression."""

import datetime
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALS = [-1.645, -0.674, 0.0, 0.674, 1.645]

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def round2(x: float) -> float | int:
    """Round x half-up to 2 decimals."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    val = whole / 100
    return int(val) if val.is_integer() else val


def parse_iso(ts: str) -> datetime.datetime:
    """Parse an ISO 8601 UTC timestamp as naive UTC."""
    return datetime.datetime.fromisoformat(ts).replace(tzinfo=None)


def read_case(path: Path) -> dict:
    """Read and parse a case JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def extract_pre23(rows: list, day_start: datetime.datetime) -> float | None:
    """Extract viewers from hourly row 1 hour before day_start."""
    for row in rows:
        ts = row.get("from")
        if ts and (parse_iso(ts) - day_start).total_seconds() == -3600:
            v = row.get("viewers")
            if v is not None and float(v) > 0:
                return float(v)
    return None


def build_hourly_series(comp: dict) -> tuple[dict[int, float], float | None]:
    """Build release-day hourly series and extract pre23 viewers."""
    day_start_raw = comp.get("day_start")
    if not day_start_raw:
        return {}, None
    day_start = parse_iso(day_start_raw)
    series: dict[int, float] = {}
    pre23: float | None = None
    for row in comp.get("hourly") or []:
        ts = row.get("from")
        if not ts:
            continue
        diff = (parse_iso(ts) - day_start).total_seconds()
        v = row.get("viewers")
        if v is not None and float(v) > 0:
            if diff == -3600:
                pre23 = float(v)
            elif 0 <= diff < 86400:
                series[1 + int(diff // 3600)] = float(v)
    return series, pre23


def collect_comparables(
    case_data: dict,
) -> tuple[float, float | None, list[tuple[dict[int, float], float | None]], list[dict[int, float]]]:
    """Filter comparables and extract target prints."""
    game_name = case_data.get("game", {}).get("name", "")
    known = case_data.get("known") or []
    h1_row = next((k for k in known if k.get("hour") == 1), None)
    p1 = (
        max(1.0, float(h1_row["viewers"]))
        if h1_row and h1_row.get("viewers") is not None
        else 1.0
    )
    ds_raw = case_data.get("day_start")
    target_pre23 = (
        extract_pre23(case_data.get("history_hourly") or [], parse_iso(ds_raw))
        if ds_raw
        else None
    )

    primary: list[tuple[dict[int, float], float | None]] = []
    fallback: list[dict[int, float]] = []

    for comp in case_data.get("comparables") or []:
        if comp.get("name") == game_name:
            continue
        u = (comp.get("unlock") or "").lower()
        if "not a launch" in u:
            continue
        skip_pri = "early access" in u or (
            "simultaneous" in u and not ("midnight local" in u or "rolling" in u)
        )
        series, comp_pre23 = build_hourly_series(comp)
        h1 = series.get(1)
        if h1 is None or h1 < 5000:
            continue
        fallback.append(series)
        if not skip_pri:
            primary.append((series, comp_pre23))

    return p1, target_pre23, primary, fallback


def solve_ols_3(xs: list[list[float]], ys: list[float]) -> list[float] | None:
    """Solve 3-feature OLS normal equations with 1e-9 ridge and partial pivoting."""
    n = len(ys)
    a = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for i in range(n):
        for j in range(3):
            b[j] += xs[i][j] * ys[i]
            for k in range(3):
                a[j][k] += xs[i][j] * xs[i][k]
    for j in range(3):
        a[j][j] += 1e-9

    aug = [a[i] + [b[i]] for i in range(3)]
    for col in range(3):
        p_row = max(range(col, 3), key=lambda r: abs(aug[r][col]))
        if abs(aug[p_row][col]) < 1e-12:
            return None
        aug[col], aug[p_row] = aug[p_row], aug[col]
        p_val = aug[col][col]
        for r in range(col + 1, 3):
            factor = aug[r][col] / p_val
            for c in range(col, 4):
                aug[r][c] -= factor * aug[col][c]

    beta = [0.0] * 3
    for r in range(2, -1, -1):
        if abs(aug[r][r]) < 1e-12:
            return None
        s = aug[r][3] - sum(aug[r][c] * beta[c] for c in range(r + 1, 3))
        beta[r] = s / aug[r][r]
    return beta


def forecast_hour(
    h: int,
    p1: float,
    target_pre23: float | None,
    primary: list[tuple[dict[int, float], float | None]],
    fallback: list[dict[int, float]],
) -> list[float | int]:
    """Forecast quantiles for hour h using 3-feature OLS or fallback."""
    usable = [(s, pre) for s, pre in primary if h in s]
    fit_ok = False
    if len(usable) >= 6:
        xs: list[list[float]] = []
        ys: list[float] = []
        for s, pre in usable:
            c_feat = (
                math.log(pre / s[1])
                if pre is not None and pre > 0
                else 0.0
            )
            xs.append([1.0, math.log(s[1]), c_feat])
            ys.append(math.log(s[h] / s[1]))
        beta = solve_ols_3(xs, ys)
        if beta is not None:
            n = len(ys)
            res = [
                y - (beta[0] + beta[1] * x[1] + beta[2] * x[2])
                for x, y in zip(xs, ys)
            ]
            std_res = math.sqrt(sum(e**2 for e in res) / n)
            med_y = statistics.median(ys)
            x1_vals = [x[1] for x in xs]
            x_min, x_max = min(x1_vals), max(x1_vals)
            log_p1 = math.log(p1)
            clamped_x = max(x_min, min(x_max, log_p1))
            t_feat = (
                math.log(target_pre23 / p1)
                if target_pre23 is not None and target_pre23 > 0
                else 0.0
            )
            fit_val = beta[0] + beta[1] * clamped_x + beta[2] * t_feat
            pred = (
                0.70 * fit_val + 0.30 * med_y
                if (log_p1 < x_min or log_p1 > x_max)
                else fit_val
            )
            pred_lr = max(min(ys), min(max(ys), pred))
            spread = max(0.25, min(0.9, std_res * 0.85))
            fit_ok = True

    if not fit_ok:
        fb = [s for s in fallback if h in s]
        if fb:
            pred_lr = statistics.median([math.log(s[h] / s[1]) for s in fb])
            spread = 0.5
        else:
            pred_lr = 0.0
            spread = 0.15 * math.sqrt(h - 1)

    qs = [round2(p1 * math.exp(pred_lr + z * spread)) for z in Z_VALS]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return [max(0.01, min(1000000000.0, q)) for q in qs]


def forecast_case(case_data: dict) -> dict:
    """Generate the full 23-hour forecast for a case.

    Iterates through hours 2 to 24 in strictly increasing order, applying the
    pre23 ratio feature and 0.85 residual spread shrinkage.
    """
    p1, target_pre23, primary, fallback = collect_comparables(case_data)
    req_hours = sorted(
        {
            r.get("hour")
            for r in case_data.get("forecast", [])
            if isinstance(r.get("hour"), int)
        }
    )
    target_hours = (
        req_hours if req_hours == list(range(2, 25)) else list(range(2, 25))
    )
    hours = [
        {
            "hour": h,
            "q": forecast_hour(h, p1, target_pre23, primary, fallback),
        }
        for h in target_hours
    ]
    hours.sort(key=lambda r: r["hour"])
    if len(hours) != 23:
        raise ValueError(f"Expected exactly 23 forecast hours, got {len(hours)}")
    return {"hours": hours}


def main() -> None:
    """Driver reading cases from INPUT_DIR and writing forecasts to OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case_data = read_case(path)
        forecast = forecast_case(case_data)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
