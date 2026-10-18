# Variable-delay timing

This shared evaluation generator makes the lead/lag challenge harder without
changing a saved encoder or the original benchmark. Each source has a period
drawn uniformly from [12,20) and an absolute delay from [0.5,1.5), instead of the
original fixed period 16 and delay 2. Noise standard deviation stays 0.005;
natural coordinate missingness stays 10%. Inputs remain C3/H32/F3.

`make_variable_delay_timing_development` returns source-disjoint TRAIN and
VALIDATION datasets under `variable-delay-timing-v1`. Each source contributes
both signs of the delay. The paired examples share period, delay, nuisance noise
and masks, and their order is randomized. Hidden storage is zero. The returned
`clean` fields contain clones of legal observations only; true period, delay and
hidden signals are not exported. Local random streams leave Torch's runtime
random state unchanged. No encoder implementation is imported and no TEST split
is generated.

`variable_delay_timing_solvability` is a separate analytic information check.
It accepts observed values and masks only, and averages
`x1(t)*x0(t+1) - x0(t)*x1(t+1)` over fully observed adjacent tuples. Features
are averaged within each time position before averaging over times. At least
four distinct supported positions and a nonzero margin are required; otherwise
the check abstains. No labels, true period/delay or fitted assets enter it.
Its noiseless sign follows the label for this recipe. Noise and missingness can
reduce accuracy or coverage, which must be reported without dropping rows.

Run the artificial data-only fixtures from the repository root in the existing
managed development container:

```powershell
.\container.ps1 -Action exec -Command @('env','PYTHONDONTWRITEBYTECODE=1','bash','code/scripts/task.sh','variable-delay-timing','-f','code/evaluation/benchmarks/variable_delay_timing/Makefile','-j1','test-variable-delay-timing')
```

The fixtures passed on 9 October 2026. They cover deterministic generation,
source isolation, paired masks, zero hidden storage, unchanged Torch random
state, invalid inputs, analytic signs for 54 noiseless cases, abstention and
hidden-NaN independence. The saved log is
`output/runs/variable-delay-timing/data-fixtures-F3PUUS/build-and-tests.log`.
They perform no encoder training, classifier fitting, PCA or quality evaluation.

The nested Makefile keeps these sources outside the frozen confirmation build
closure. Before measuring encoders here, freeze a separate prospective card,
cohort seeds and saved-evidence reader as described in the
[benchmark plan](../../../../doc/HARDER_TIMING_BENCHMARK_PLAN.md).
