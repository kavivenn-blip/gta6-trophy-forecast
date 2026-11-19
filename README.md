# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kavivenn-blip/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kavivenn-blip/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast of every GTA VI PS5 trophy's Exophase earn rate one week after launch, as five quantiles per trophy (0.05/0.25/0.5/0.75/0.95), scored by mean pinball loss on the logit scale.

## Method — two models, combined a priori

**M1 — logit trend.** Per trophy, logit(rate) is regressed on ln(players tracked) and ln(1 + hours), fit on the last 24 hours of reads. The players term carries dilution (a refresh brings in new players who own fewer trophies, so rates fall); the time term carries progression (between refreshes the tracked cohort plays further, so rates creep up). Slopes are shrunk toward their tier's fit (partial pooling), because rare trophies sit near the 2-decimal display floor and their own slopes are noisy. The players count at the target comes from a damped power-law fit anchored on the last read.

**M3 — cohort simulation.** The process, not the curve: each trophy's holders follow a logistic in ln(1 + hours) — few holders at launch, fast arrival mid-week, levelling off — with its ceiling profiled over a grid, and the rate at the target is holders divided by the projected players. Dilution and progression are explicit instead of tangled in one regression.

**Combine.** The two models' per-level quantiles are combined by their median on the logit scale (with two models, the mean of the two logit quantiles), decided before any result was seen — not picked after the fact. Spreads come from each fit's residual variance times its prediction leverage at the target, plus the players-projection uncertainty and a prior on how wrong a fitted log-time slope can be over the extrapolation gap. Both models nest on the house persistence baseline when the history is flat.

**Guards.** The (near-)everyone trophy stays at 99.99%; the platinum median is capped at 1% (only the fastest completionists hold it a week in); the forecast is for the rate the Exophase page will *show* at the target, including the chance the page has not refreshed by then.

## Validation (no cheap test was possible)

The practice answers are sealed and the final history is not published before the deadline, so no backtest against real answers exists — the build was approved without the cheap test, and this section says exactly what was checked instead:

- **Practice cases p00/p01:** output valid (55 rows, quantiles non-decreasing, 0–100, 2,679 bytes), two runs byte-identical, 0.1 s per run; on the flat launch-night history both models degenerate to persistence, and the medians match the house starter exactly.
- **Synthetic final-like case:** a self-built case with a known trajectory (360 reads over 60 h, players growing with refresh jumps, holders following per-trophy logistics) scored against its own truth — pinball 0.084 for this entry vs 0.32 for persistence and 0.33 for the house curve, median logit error 0.11 vs 0.52. Synthetic, not real: it proves the code paths work and the models beat the house baselines *when the trajectory has the assumed shape*; it says nothing about the real final draw.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. The program reads every `*.in` case in the input directory and writes `<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`) to the output directory. Deterministic: two runs write identical bytes, well inside the 8-second limit.

---

<sub>Entered on [make.fun](https://make.fun/?ref=hp1z0q8b&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
