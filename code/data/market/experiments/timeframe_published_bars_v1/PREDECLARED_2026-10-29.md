# Timeframe search, declared before the runs

The search is closed. Outcomes are in [CLOSE_2026-10-29.md](CLOSE_2026-10-29.md).

Baseline is the finished width-128 instance, seed 949102, history 128, temporal
weight 0.1. Its validation R² values are anchor volatility 0.7682422126010569,
trend 0.22475307149870904, activity 0.9955882717924301. Cross resolution
abstained. These runs do not replace seed 946103 and do not call `decision.mjs`.

Closed and not retried: four layers, mask 0.15, mask 0.35, learning rate
0.0005, eight heads, feed-forward 512.

## temporal1

Same windows, settings, and budget as width 128. One change: temporal loss
weight 1 instead of the frozen 0.1. Seed 949201, row 950201, mask 951201,
context 952201. Batch 8, 2048 updates.

A keep, on this one seed, needs trend R² at least 0.24475307149870904,
volatility at least 0.75, and activity at least 0.99. Anything else closes
weight 1. No further weight sweep follows in this round.

## history480

Same width-128 capacity and temporal weight 0.1. History becomes 480 minutes
so the hourly channel can supply eight bars. Daily bars still abstain.
Validation anchors shrink until the 480-minute windows do not overlap.
Seed 949301, row 950301, mask 951301, context 952301. Batch 8, falling back
to batch 4 only if CUDA reports out of memory.

Its anchor R² values are a longer-window measurement. They are not a ranking
against the history-128 numbers. The keep for this version is a supported
`cross_resolution` score. Abstention closes history 480. No longer history
follows in this round.

## Results of that round

Weight 1 reached trend 0.3394718844839739, volatility 0.7242716221847898,
activity 0.9940847992237186. Trend and activity cleared the keep. Volatility
missed 0.75, so weight 1 is closed. History 480 made cross resolution
measurable on all 48 validation rows at R² -0.23302912246862673. The hourly
agreement is not linearly carried. No longer history follows.

## temporal03

Declared after those two results. Same 128-minute windows, width 128, and
2048-update budget. One change from the baseline: temporal weight 0.3.
Seed 949202, row 950202, mask 951202, context 952202.

A keep needs trend R² at least 0.24475307149870904, volatility at least 0.75,
and activity at least 0.99. That keeps the direction gain without the
volatility spend measured at weight 1. A miss closes the weight axis for this
round. Weight 0.3 does not replace seed 946103.

Weight 0.3, seed 949202, scored trend 0.04862204088539579, volatility
0.7698685401640426, activity 0.9955803573375589. Volatility and activity
held. Trend missed, so no other coefficient is opened.

## temporal1 repeat

The weight-1 trend gain is one seed. Repeat that same run with seed 949203,
row 950203, mask 951203, context 952203. Same windows, weight 1, 2048 updates.

The gain is repeatable only if this seed also reaches trend R² at least 0.28.
A lower trend closes weight 1 as a one-seed result. This repeat does not
replace seed 946103.

Seed 949203 scored trend 0.2328108813893348, volatility 0.760405695079108,
activity 0.9936182294809958. Trend is below 0.28, so weight 1 stays a
one-seed result and the weight axis is closed. The development instance
remains width 128, seed 949102, temporal weight 0.1.
