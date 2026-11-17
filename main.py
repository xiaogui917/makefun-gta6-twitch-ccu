"""GTA VI Twitch CCU forecast: comparables log-ratio regression."""

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


def build_hourly_series(comp: dict) -> dict[int, float]:
    """Extract release-day hourly viewer counts (hours 1..24)."""
    day_start_raw = comp.get("day_start")
    if not day_start_raw:
        return {}
    day_start = parse_iso(day_start_raw)
    series: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        diff_sec = (parse_iso(row["from"]) - day_start).total_seconds()
        if 0 <= diff_sec < 86400:
            h = 1 + int(diff_sec // 3600)
            v = row.get("viewers")
            if v is not None and v > 0:
                series[h] = float(v)
    return series


def collect_comparables(
    case_data: dict,
) -> tuple[float, list[dict[int, float]], list[dict[int, float]]]:
    """Filter comparables into primary usable and fallback series."""
    game_name = case_data.get("game", {}).get("name", "")
    known = case_data.get("known") or []
    h1_row = next((k for k in known if k.get("hour") == 1), None)
    p1 = (
        max(1.0, float(h1_row["viewers"]))
        if h1_row and h1_row.get("viewers") is not None
        else 1.0
    )

    primary: list[dict[int, float]] = []
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
        series = build_hourly_series(comp)
        h1 = series.get(1)
        if h1 is None or h1 < 5000:
            continue
        fallback.append(series)
        if not skip_pri:
            primary.append(series)

    return p1, primary, fallback


def forecast_hour(
    h: int,
    p1: float,
    primary: list[dict[int, float]],
    fallback: list[dict[int, float]],
) -> list[float | int]:
    """Forecast quantiles for hour h using OLS log ratio or fallback."""
    usable = [s for s in primary if h in s]
    if len(usable) >= 6:
        xs = [math.log(s[1]) for s in usable]
        ys = [math.log(s[h] / s[1]) for s in usable]
        n = len(xs)
        x_bar = sum(xs) / n
        y_bar = sum(ys) / n
        var_x = sum((x - x_bar) ** 2 for x in xs)
        beta = (
            sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / var_x
            if var_x > 0
            else 0.0
        )
        alpha = y_bar - beta * x_bar
        res = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
        std_res = math.sqrt(sum(e**2 for e in res) / n)
        med_y = statistics.median(ys)

        x_min, x_max = min(xs), max(xs)
        log_p1 = math.log(p1)
        clamped_x = max(x_min, min(x_max, log_p1))
        fit_val = alpha + beta * clamped_x
        pred = (
            0.70 * fit_val + 0.30 * med_y
            if (log_p1 < x_min or log_p1 > x_max)
            else fit_val
        )
        pred_lr = max(min(ys), min(max(ys), pred))
        spread = max(0.25, min(0.9, std_res))
    else:
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

    Robustly ensures all hours 2 through 24 are evaluated in strictly increasing order,
    matching required forecast hour targets while guarding against omissions.
    """
    p1, primary, fallback = collect_comparables(case_data)
    req_hours = sorted({r.get("hour") for r in case_data.get("forecast", []) if isinstance(r.get("hour"), int)})
    target_hours = req_hours if req_hours == list(range(2, 25)) else list(range(2, 25))
    hours = [
        {"hour": h, "q": forecast_hour(h, p1, primary, fallback)}
        for h in target_hours
    ]
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
