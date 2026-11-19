"""GTA VI trophy rarity forecast — trend extrapolation plus cohort simulation.

Entry for make.fun gta6-trophy-rarity-forecast-01: forecast every PS5 trophy's
Exophase earn rate at the case target as five quantiles (0.05/0.25/0.5/0.75/0.95),
scored by mean pinball loss on the logit scale.

Two models, combined a priori by taking their per-level median (with two models
that is the mean of the two logit quantiles):

M1 — logit trend. For each trophy, regress logit(rate) on ln(players_tracked)
and ln(1 + hours since the first read), fit on the last 24 hours of reads:
the recent trend is the one that extrapolates, while the whole-history
average slope points the wrong way once the trajectory bends (the house
curve fits the last 24 hours for the same reason). The first term carries
dilution: when a refresh brings in a wave of new players who own fewer
trophies, rates fall. The second carries progression: between refreshes
the tracked cohort plays further and rates creep up. Slopes are shrunk
toward their tier's fit (partial pooling), because rare trophies sit near
the 2-decimal display floor and their own slopes are noisy. The players
count at the target comes from a damped power-law fit to its own history,
anchored on the last read.

M3 — cohort simulation. Model the process, not the curve: holders(t)
follow a logistic in ln(1 + hours) per trophy (a saturating S-curve: few
holders at launch, fast arrival mid-week, levelling off), players(t) a
damped power law of its own, and the rate at the target is
holders/players. Dilution and progression are explicit instead of tangled
in one regression. The logistic's ceiling is profiled over a grid, so a
trophy with no saturation seen yet nests the power law it outgrows.

Spreads come from each fit's residual variance times its prediction leverage
at the target, plus the players-projection uncertainty — so a trophy whose
history cannot pin down its slope gets an honestly wide interval, and one
with a clean trajectory gets a narrow one. Both models nest on the house
persistence baseline when the history is flat (no usable slope), and every
fallback is persistence, never a crash.

Guards: the 100% trophy stays ~100%; the platinum median is capped at 1%
(only the fastest completionists hold it a week in); output is 55 rows,
quantiles non-decreasing, 0..100, at most 65,536 bytes, byte-identical
across runs (no clock, no network, no randomness, fixed file order).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves this file's directory OFF sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()
Z = [_NORM.inv_cdf(p) for p in LEVELS]

# Spread of the house persistence baseline, logit per sqrt-day: the floor
# every fallback uses, and the scale our own spreads are measured against.
HOUSE_SIGMA = 0.5
# How much of the fitted players-growth exponent survives extrapolation:
# launch growth saturates, so the raw power law would overshoot the target.
GROWTH_DAMP = 0.7
# Prior on how wrong a fitted log-time slope can be, per unit of log-time
# extrapolated past the last read: the trend itself is uncertain, and the
# residual variance alone cannot see that (a noiseless history still
# extrapolates). Added in quadrature to the spreads.
TREND_SE = 0.25
# Hard cap on the projected players count, as a multiple of the last read.
GROWTH_CAP = 5.0
# Logit uncertainty of the players projection itself.
PLAYERS_SE = 0.2
# Platinum: a week in, only the fastest completionists hold it.
PLAT_MEDIAN_CAP = 1.0
PLAT_Q75_CAP = 1.5
PLAT_Q95_CAP = 2.0
# A trophy the last read shows at (nearly) everyone keeps that rate.
NEARLY_ALL = 99.5
ALL_RATE = 99.99


def clamp_pct(x):
    """A percent into the [0.01, 99.99] range the logit is defined on."""
    return min(ALL_RATE, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as the house baseline's rounding does.

    A whole number comes back as an int, so the JSON writes 100 and not 100.0.
    """
    d = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return int(d) if d == d.to_integral_value() else float(d)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _solve(matrix, vec):
    """Solve a small dense system by Gauss-Jordan with partial pivoting.

    Returns the solution, or None when the matrix is (near-)singular.
    """
    n = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def _invert(matrix):
    n = len(matrix)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        sol = _solve(matrix, e)
        if sol is None:
            return None
        cols.append(sol)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit_ridge(rows, y, lam):
    """Least squares with a ridge on the slope terms.

    rows: [1.0, x1, x2, ...] per observation (first column is the intercept).
    Slope columns are standardised before the ridge, so one penalty scale
    fits every regressor; a slope column with (near-)zero variance is
    dropped outright (its coefficient is 0) instead of being left to fight
    the intercept for the level. Returns (beta, inv, meta) with beta in the
    original coordinates, inv the inverse of the standardised (X'X + lam*I)
    matrix, and meta the (kept columns, means, stds) needed to standardise
    a new point for its leverage. Returns None when the system is singular.
    """
    k = len(rows[0])
    n = len(rows)
    means = [0.0] * k
    stds = [1.0] * k
    keep = [True] * k
    for j in range(1, k):
        vals = [r[j] for r in rows]
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / n
        s = math.sqrt(var)
        means[j] = m
        if s < 1e-12:
            keep[j] = False
        else:
            stds[j] = s
    idx = [j for j in range(k) if keep[j]]
    m = len(idx)
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    for r, yv in zip(rows, y):
        z = [1.0 if j == 0 else (r[j] - means[j]) / stds[j] for j in idx]
        for a in range(m):
            xty[a] += z[a] * yv
            for b in range(m):
                xtx[a][b] += z[a] * z[b]
    for a in range(1, m):
        xtx[a][a] += lam
    inv = _invert(xtx)
    beta_z = _solve(xtx, xty)
    if beta_z is None or inv is None:
        return None
    beta = [0.0] * k
    for a, j in enumerate(idx):
        if j != 0:
            beta[j] = beta_z[a] / stds[j]
    beta[0] = beta_z[0] - sum(
        beta_z[a] * means[j] / stds[j] for a, j in enumerate(idx) if j != 0
    )
    return beta, inv, (idx, means, stds)


def leverage(inv, meta, point):
    """x0' (X'X + lam*I)^-1 x0 for a new point, in standardised space."""
    idx, means, stds = meta
    z = [1.0 if j == 0 else (point[j] - means[j]) / stds[j] for j in idx]
    return sum(
        z[a] * inv[a][b] * z[b] for a in range(len(z)) for b in range(len(z))
    )


def predict(beta, point):
    return sum(b * x for b, x in zip(beta, point))


def project_players(reads, h_target):
    """Players tracked at the target, from a damped power-law fit.

    ln(P) is fit on ln(1 + hours) over the last three quarters of the
    history (the current growth regime, not launch night). The projection
    is anchored on the last read — only the INCREMENT is damped, because
    launch growth saturates — and capped at GROWTH_CAP times the last
    read, floored at the last read. Returns (players_target,
    players_vary): the second flag says whether the history's players
    count moved at all, which decides whether the projection uncertainty
    enters the spreads.
    """
    n = len(reads)
    if n == 0:
        return None, False
    last_p = reads[-1]["players_tracked"]
    ln_p_last = math.log(max(1.0, float(last_p)))
    if n < 3:
        return float(last_p), False
    xs = [math.log1p(r["h"]) for r in reads]
    lps = [math.log(max(1.0, float(r["players_tracked"]))) for r in reads]
    if max(lps) - min(lps) < 1e-9:
        return float(last_p), False
    cut = max(0, n - (3 * n) // 4)
    rows = [[1.0, xs[i]] for i in range(cut, n)]
    ys = lps[cut:]
    fit = fit_ridge(rows, ys, lam=0.02 * len(rows))
    if fit is None:
        return float(last_p), True
    (_, v), _, _ = fit
    v = min(max(v, 0.0), 2.0)
    ln_p = ln_p_last + GROWTH_DAMP * v * (math.log1p(h_target) - xs[-1])
    p_target = math.exp(ln_p)
    p_target = min(p_target, GROWTH_CAP * last_p)
    return max(p_target, float(last_p)), True


def quantiles_from(median_logit, sigma):
    """Five quantiles around a logit median with a logit spread."""
    return [inv_logit_pct(median_logit + z * sigma) for z in Z]


def persistence_quantiles(last_rate, horizon_days):
    """The house baseline: last rate, random-walk spread."""
    return quantiles_from(logit_pct(last_rate), HOUSE_SIGMA * math.sqrt(horizon_days))


def fit_m1(m1_series, tier_betas, h_target, ln_p_target, players_vary, trend_gap):
    """M1: logit(rate) ~ ln(players) + ln(1 + hours), per trophy.

    m1_series: per-trophy list of (h, ln_p, logit_rate) per read with a
    rate. tier_betas: per-trophy tier fit [a, b, c] or None, for partial
    pooling of the slopes (not the level). trend_gap: how far past the
    last read the target sits, in ln(1+h) — the trend itself is uncertain
    over that gap. Returns a list of (median_logit, sigma) per trophy
    index; None where the fit cannot pin the trophy down (the caller
    falls back to persistence).
    """
    out = []
    for i, series in enumerate(m1_series):
        if len(series) < 3:
            out.append(None)
            continue
        rows = [[1.0, lp, math.log1p(h)] for h, lp, _ in series]
        ys = [y for _, _, y in series]
        fit = fit_ridge(rows, ys, lam=0.05 * len(rows))
        if fit is None:
            out.append(None)
            continue
        beta, inv, meta = fit
        tier_beta = tier_betas[i]
        if tier_beta is not None:
            w = len(series) / (len(series) + 30.0)
            for j in (1, 2):
                beta[j] = w * beta[j] + (1.0 - w) * tier_beta[j]
        resid = [y - predict(beta, r) for r, y in zip(rows, ys)]
        dof = max(1, len(rows) - 3)
        s2 = sum(e * e for e in resid) / dof
        point = [1.0, ln_p_target, math.log1p(h_target)]
        med = predict(beta, point)
        lev = max(0.0, leverage(inv, meta, point))
        var = s2 * (1.0 + lev)
        if players_vary:
            var += (beta[1] * PLAYERS_SE) ** 2
        var += (TREND_SE * trend_gap) ** 2
        sigma = min(1.6, max(0.12, math.sqrt(max(var, 0.0))))
        out.append((med, sigma))
    return out


def fit_logistic_holders(points):
    """Fit H = C / (1 + exp(-(a + b*ln(1+h)))) by profiling C over a grid.

    points: list of (h, holders > 0). For each candidate C the logit of
    H/C is linear in ln(1+h), so (a, b) come from an ordinary fit; C is
    picked by the residual variance of ln H, the scale holders live on.
    A very large best C means "no saturation seen yet" — the logistic
    then behaves like the power law it nests. Returns (C, a, b, s2, inv,
    meta) with s2 the ln-H residual variance and inv/meta from the linear
    fit at the best C, for leverage; None when nothing pins a fit down.
    """
    if len(points) < 3:
        return None
    h_max = max(hv for _, hv in points)
    best = None
    for mult in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0):
        c = mult * h_max
        rows, ys = [], []
        for h, hv in points:
            if hv <= 0 or hv >= 0.999 * c:
                continue
            rows.append([1.0, math.log1p(h)])
            ys.append(math.log(hv / (c - hv)))
        if len(rows) < 3:
            continue
        fit = fit_ridge(rows, ys, lam=0.02 * len(rows))
        if fit is None:
            continue
        (a, b), _, _ = fit
        sse, cnt = 0.0, 0
        for h, hv in points:
            pred = c / (1.0 + math.exp(-(a + b * math.log1p(h))))
            if pred > 0 and hv > 0:
                sse += (math.log(hv) - math.log(pred)) ** 2
                cnt += 1
        if cnt < 3:
            continue
        score = sse / max(1, cnt - 3)
        if best is None or score < best[0]:
            best = (score, c, a, b, rows, ys)
    if best is None:
        return None
    _, c, a, b, rows, ys = best
    b = min(max(b, 0.0), 3.0)
    a = sum(y - b * r[1] for r, y in zip(rows, ys)) / len(rows)
    sse = 0.0
    for h, hv in points:
        pred = c / (1.0 + math.exp(-(a + b * math.log1p(h))))
        if pred > 0 and hv > 0:
            sse += (math.log(hv) - math.log(pred)) ** 2
    s2 = sse / max(1, len(rows) - 2)
    fit = fit_ridge(rows, ys, lam=0.02 * len(rows))
    if fit is None:
        return None
    _, inv, meta = fit
    return c, a, b, s2, inv, meta


def fit_m3(m3_series, tier_bs, h_target, p_target, players_vary, trend_gap):
    """M3: holders follow a logistic in ln(1 + hours); rate = holders / players.

    m3_series: per-trophy list of (h, holders) per read with holders > 0.
    tier_bs: per-trophy tier holder slope or None, for partial pooling.
    trend_gap: how far past the last read the target sits, in ln(1+h).
    Returns a list of (median_logit, sigma) per trophy index; None where
    the fit cannot pin the trophy down.
    """
    out = []
    for i, series in enumerate(m3_series):
        got = fit_logistic_holders(series)
        if got is None:
            out.append(None)
            continue
        c, a, b, s2, inv, meta = got
        tier_b = tier_bs[i]
        if tier_b is not None:
            w = len(series) / (len(series) + 30.0)
            b = w * b + (1.0 - w) * tier_b
            b = min(max(b, 0.0), 3.0)
            rows = [[1.0, math.log1p(h)] for h, hv in series if 0 < hv < 0.999 * c]
            ys = [math.log(hv / (c - hv)) for h, hv in series if 0 < hv < 0.999 * c]
            if len(rows) >= 3:
                a = sum(y - b * r[1] for r, y in zip(rows, ys)) / len(rows)
        h_t = c / (1.0 + math.exp(-(a + b * math.log1p(h_target))))
        rate = 100.0 * h_t / p_target
        med = logit_pct(rate)
        lev = max(0.0, leverage(inv, meta, [1.0, math.log1p(h_target)]))
        var = s2 * (1.0 + lev)
        if players_vary:
            var += PLAYERS_SE ** 2
        var += (TREND_SE * trend_gap) ** 2
        sigma = min(1.6, max(0.15, math.sqrt(max(var, 0.0))))
        out.append((med, sigma))
    return out


def tier_fits(reads, reads_m1, trophies):
    """Per-tier reference fits, for partial pooling of the slopes.

    reads: the full history (M3's logistic needs the whole rise to pin
    its ceiling). reads_m1: the last-24h window (M1's trend is the recent
    one — the whole-history average slope points the wrong way once the
    trajectory bends, which is exactly why the house curve fits the last
    24 hours). Returns (tier_index, m1_series, tier_beta, m3_series,
    tier_g): tier_index maps a trophy index to its tier key; m1_series /
    m3_series hold the per-trophy series M1/M3 fit on; tier_beta is the
    tier's M1 slopes [a, b, c] (or None) and tier_g the tier's M3 holder
    slope.
    """
    tiers = {}
    for i, t in enumerate(trophies):
        tiers.setdefault(t.get("tier", "bronze"), []).append(i)
    tier_index = {}
    for key, members in tiers.items():
        for i in members:
            tier_index[i] = key
    # Per-trophy series, built once and shared by M1 and M3.
    m1_series = []
    m3_series = []
    for i in range(len(trophies)):
        s1, s3 = [], []
        for r in reads_m1:
            rate = r["rates"][i]
            if rate is None:
                continue
            s1.append((r["h"], r["lp"], logit_pct(rate)))
        for r in reads:
            rate = r["rates"][i]
            if rate is None:
                continue
            holders = rate / 100.0 * r["players_tracked"]
            if holders > 0:
                s3.append((r["h"], holders))
        m1_series.append(s1)
        m3_series.append(s3)
    # Tier-level fits on the tier-mean logit series and holder series.
    tier_beta = {}
    tier_g = {}
    for key, members in tiers.items():
        rows, ys = [], []
        for r in reads_m1:
            vals = [logit_pct(r["rates"][i]) for i in members if r["rates"][i] is not None]
            if vals:
                rows.append([1.0, r["lp"], math.log1p(r["h"])])
                ys.append(sum(vals) / len(vals))
        if len(rows) >= 3:
            fit = fit_ridge(rows, ys, lam=0.05 * len(rows))
            tier_beta[key] = fit[0] if fit else None
        else:
            tier_beta[key] = None
        rows3 = []
        for r in reads:
            hs = [
                rate / 100.0 * r["players_tracked"]
                for i in members
                if (rate := r["rates"][i]) is not None and rate > 0
            ]
            if hs:
                rows3.append((r["h"], sum(hs) / len(hs)))
        got3 = fit_logistic_holders(rows3)
        tier_g[key] = got3[2] if got3 else None
    return tier_index, m1_series, tier_beta, m3_series, tier_g


def cap_platinum(qs, tier):
    """Platinum guardrails: the median stays at or below 1% a week in."""
    if tier != "platinum":
        return qs
    # Caps per level: q05/q25 may sit anywhere below 2.0, q50 <= 1.0,
    # q75 <= 1.5, q95 <= 2.0.
    caps = [2.0, 2.0, PLAT_MEDIAN_CAP, PLAT_Q75_CAP, PLAT_Q95_CAP]
    return [min(q, c) for q, c in zip(qs, caps)]


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    n_tr = len(trophies)
    if not history:
        # No reads at all: no signal. Emit a valid, neutral forecast
        # rather than crashing the case.
        rows = []
        for t in trophies:
            qs = quantiles_from(logit_pct(1.0), HOUSE_SIGMA * math.sqrt(horizon_days))
            qs = cap_platinum(qs, t.get("tier", "bronze"))
            rows.append({"id": t["id"], "q": [round4(q) for q in qs]})
        return {"trophies": rows}
    t0 = parse_time(history[0]["at"])
    target_t = parse_time(case["target"])
    h_target = max(0.0, (target_t - t0) / 3600.0)
    reads = []
    for r in history:
        at = parse_time(r["at"])
        reads.append(
            {
                "h": max(0.0, (at - t0) / 3600.0),
                "players_tracked": r["players_tracked"],
                "rates": r["rates"],
                "lp": math.log(max(1.0, float(r["players_tracked"]))),
            }
        )
    p_target, players_vary = project_players(reads, h_target)
    if p_target is None:
        p_target = float(reads[-1]["players_tracked"])
    ln_p_target = math.log(max(1.0, p_target))
    # M1 fits the last 24 hours of reads (the recent trend); with fewer
    # than 12 reads in the window it takes the whole history instead.
    h_last = reads[-1]["h"]
    windowed = [r for r in reads if r["h"] >= h_last - 24.0]
    reads_m1 = windowed if len(windowed) >= 12 else reads
    trend_gap = max(0.0, math.log1p(h_target) - math.log1p(h_last))
    tier_index, m1_series, tier_beta, m3_series, tier_g = tier_fits(reads, reads_m1, trophies)
    # M1 and M3 each get their tier's reference fit for partial pooling.
    tier_betas = [tier_beta[tier_index[i]] for i in range(n_tr)]
    tier_gs = [tier_g[tier_index[i]] for i in range(n_tr)]
    m1_out = fit_m1(m1_series, tier_betas, h_target, ln_p_target, players_vary, trend_gap)
    m3_out = fit_m3(m3_series, tier_gs, h_target, p_target, players_vary, trend_gap)
    rows = []
    for i, t in enumerate(trophies):
        last = next(
            (r["rates"][i] for r in reversed(reads) if r["rates"][i] is not None),
            None,
        )
        if last is None:
            last = 1.0
        tier = t.get("tier", "bronze")
        if last >= NEARLY_ALL:
            # The (near-)everyone trophy: the page will still show ~100%.
            rows.append({"id": t["id"], "q": [ALL_RATE] * 5})
            continue
        persist = persistence_quantiles(last, horizon_days)
        if m1_out[i] is None:
            q1 = persist
        else:
            med, sig = m1_out[i]
            q1 = quantiles_from(med, sig)
        if m3_out[i] is None:
            q3 = persist
        else:
            med, sig = m3_out[i]
            q3 = quantiles_from(med, sig)
        # A priori median of the two models, per level, on the logit scale.
        combined = [
            inv_logit_pct((logit_pct(a) + logit_pct(b)) / 2.0)
            for a, b in zip(q1, q3)
        ]
        combined = cap_platinum(combined, tier)
        # Keep the quantiles non-decreasing after the caps.
        mono = []
        run = 0.0
        for q in combined:
            run = max(run, q)
            mono.append(run)
        rows.append({"id": t["id"], "q": [round4(q) for q in mono]})
    return {"trophies": rows}


def main():
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
