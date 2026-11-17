# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alro07/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alro07/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, as five quantiles per hour, from the first hour's viewer count and the release-day curves of past big game launches.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=q57bcks6&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## Methodology

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
every `*.in` case in `$MAKEFUN_INPUT_DIR` and writes `<id>.out` with five
quantiles (5%, 25%, 50%, 75%, 95%) for hours 2–24.

The forecast works in log space, anchored on the case's own hour-1 print:

- For each comparable past launch, it takes that launch's release-day hourly
  curve and forms log-ratios `log(viewers_h / viewers_1)`, hours 2–24.
- For each hour it fits a small weighted ridge regression of the log-ratio on
  an intercept, `log10(hour-1 viewers)` centred at 100k, and a console-only
  flag. Across the comparables, the bigger hour 1 was, the flatter the rest
  of the day ran; console-only launches run a higher evening peak. Launches
  are weighted by how close their hour-1 scale is to this case's (Gaussian
  kernel on log10 viewers), boosted for console-only launches and rolling
  midnight-local unlocks, discounted for early-access-contaminated days.
- The fitted curve is shrunk 15% toward the weighted median curve and lightly
  smoothed across neighbouring hours. The case's own game (the GTA VI
  pre-release history) is never used as a launch.
- Spreads come from the weighted spread of the regression residuals per hour,
  floored at `0.20 + 0.03·sqrt(h−1)` log units, so the band widens over the
  day; quantiles are `median · exp(z_tau · s_h)`.

Validation (leave-one-out over the 37 usable comparables in the practice
cases, scored with the contest's own pinball loss on the log scale):

| model | all launches | hour-1 ≥ 50k only |
|---|---:|---:|
| house persistence baseline | 0.930 | 0.251 |
| this entry | 0.443 | 0.203 |

Typical miss on big launches is ~40% with 90% coverage ~0.94, against the
baseline's ~83% miss. The pre-release trend was tested and does not predict
the day's shape, so it is used only to guard a degenerate hour-1 print.
