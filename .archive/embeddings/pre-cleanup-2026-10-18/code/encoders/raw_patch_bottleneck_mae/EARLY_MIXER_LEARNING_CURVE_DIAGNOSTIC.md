# Early mixer fresh learning-curve diagnostic

Protocol: `early-mixer-learning-curve-v1`. Completed fresh continuous CUDA training and known-VALIDATION scoring, with an independent saved-arithmetic audit. No TEST or stress was generated or read. All declared budgets are retained; no best point is selected.

RPB-v7.alt-03 is the fresh late-mixer group using unchanged coordinate-0.15 training. RPB-v10.alt-01 moves aligned W64 channel mixing before temporal encoding and retains its separate independent local temporal pass. Both have 225,805 parameters and serve native 32. Equal parameters and updates do not mean equal compute. These are new groups; the completed fixed-512 groups and original saved v7 instances are preserved.

Five paired timing masters are 30130/31231/32332/33433/34534. Each of the ten trajectories has one live AdamW optimizer and frozen scaler through 0/512/1024/2048. Ordinary reconstruction jointly trains encoder and decoder parameters that receive gradients. Extra decoder-only calibration updates are zero. Batch size is 8, with 256 TRAIN rows from 128 groups and 128 VALIDATION rows from 64 disjoint groups. At 2,048 updates each trajectory has 16,384 sampled row exposures: 64 equivalent presentations, not guaranteed epochs. Across all trajectories this is 163,840 exposures. Nested budgets and head repetitions are not additional independent encoder runs.

Amplitude data masters 35635/36736/37837/38938/40039 fit separate TRAIN heads and controls only. Encoder and scaler fitting uses timing TRAIN. Amplitude uses point 0 and 2,048 only; no intermediate amplitude features or heads were computed.

Three fixed repetitions 2701/2802/2903 use Ridge penalty 1 and tanh-16 Adam learning rate 0.01 for 100 updates. Native 32 has no PCA afterward. Raw 576 contains 288 scaled values and 288 masks; mask-only has 288 inputs. Raw TRAIN outer normalization is shared with standalone PCA 32; probe TRAIN normalization remains separate. Linear/neural parameter counts are 66/562 at width 32, 578/4,658 at 288 and 1,154/9,266 at 576.

Both initial controls are retained because equal initialized weights do not imply equal outputs for different computation order. Controls and their fitted heads are reused across the timing budget panels. Each positive point has its own fixed TRAIN-fitted heads, reused for intact and one extra 30% coordinate-deletion VALIDATION view. Coverage is valid/declared rows; accuracy is conditional on that support. All per-head and paired source intervals are retained in the durable JSON.

## Timing (lag sign) at 512 updates: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 47.6562 | 51.8229 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.7500 | 71.9271 | 100.0000 |
| Untrained late encoder | 32 | 53.4375 | 65.1042 | 100.0000 |
| Untrained early encoder | 32 | 52.6563 | 66.9792 | 100.0000 |
| RPB-v7.alt-03 | 32 | 94.8438 | 98.3333 | 100.0000 |
| RPB-v10.alt-01 | 32 | 92.5000 | 98.9583 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 30130 | 91.4062 / 98.1771 | 96.0938 / 98.4375 | +4.6875 | 100.0000 / 100.0000 |
| 31231 / 31231 | 100.0000 / 100.0000 | 99.2188 / 100.0000 | -0.7812 | 100.0000 / 100.0000 |
| 32332 / 32332 | 92.9688 / 94.2708 | 88.2812 / 96.6146 | -4.6875 | 100.0000 / 100.0000 |
| 33433 / 33433 | 95.3125 / 100.0000 | 95.3125 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 34534 / 34534 | 94.5312 / 99.2188 | 83.5938 / 99.7396 | -10.9375 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Timing (lag sign) at 512 updates: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 48.9062 | 49.4792 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.2812 | 52.8125 | 100.0000 |
| Untrained late encoder | 32 | 50.7812 | 52.3958 | 100.0000 |
| Untrained early encoder | 32 | 48.7500 | 52.6042 | 100.0000 |
| RPB-v7.alt-03 | 32 | 89.3750 | 92.9167 | 100.0000 |
| RPB-v10.alt-01 | 32 | 92.1875 | 96.5104 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 30130 | 85.1562 / 86.9792 | 96.0938 / 95.3125 | +10.9375 | 100.0000 / 100.0000 |
| 31231 / 31231 | 95.3125 / 95.8333 | 95.3125 / 99.2188 | +0.0000 | 100.0000 / 100.0000 |
| 32332 / 32332 | 86.7188 / 88.2812 | 86.7188 / 92.7083 | +0.0000 | 100.0000 / 100.0000 |
| 33433 / 33433 | 89.0625 / 96.0938 | 97.6562 / 99.4792 | +8.5938 | 100.0000 / 100.0000 |
| 34534 / 34534 | 90.6250 / 97.3958 | 85.1562 / 95.8333 | -5.4688 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Timing (lag sign) at 1024 updates: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 47.6562 | 51.8229 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.7500 | 71.9271 | 100.0000 |
| Untrained late encoder | 32 | 53.4375 | 65.1042 | 100.0000 |
| Untrained early encoder | 32 | 52.6563 | 66.9792 | 100.0000 |
| RPB-v7.alt-03 | 32 | 93.2812 | 98.9583 | 100.0000 |
| RPB-v10.alt-01 | 32 | 94.3750 | 98.7500 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 30130 | 92.9688 / 99.4792 | 92.9688 / 94.7917 | +0.0000 | 100.0000 / 100.0000 |
| 31231 / 31231 | 100.0000 / 99.7396 | 97.6562 / 100.0000 | -2.3438 | 100.0000 / 100.0000 |
| 32332 / 32332 | 92.9688 / 98.4375 | 95.3125 / 100.0000 | +2.3438 | 100.0000 / 100.0000 |
| 33433 / 33433 | 82.8125 / 97.6562 | 94.5312 / 100.0000 | +11.7188 | 100.0000 / 100.0000 |
| 34534 / 34534 | 97.6562 / 99.4792 | 91.4062 / 98.9583 | -6.2500 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Timing (lag sign) at 1024 updates: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 48.9062 | 49.4792 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.2812 | 52.8125 | 100.0000 |
| Untrained late encoder | 32 | 50.7812 | 52.3958 | 100.0000 |
| Untrained early encoder | 32 | 48.7500 | 52.6042 | 100.0000 |
| RPB-v7.alt-03 | 32 | 87.1875 | 94.5312 | 100.0000 |
| RPB-v10.alt-01 | 32 | 92.9688 | 96.4583 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 30130 | 84.3750 / 89.5833 | 89.0625 / 86.9792 | +4.6875 | 100.0000 / 100.0000 |
| 31231 / 31231 | 86.7188 / 89.0625 | 97.6562 / 99.4792 | +10.9375 | 100.0000 / 100.0000 |
| 32332 / 32332 | 89.8438 / 98.9583 | 93.7500 / 99.4792 | +3.9062 | 100.0000 / 100.0000 |
| 33433 / 33433 | 80.4688 / 97.3958 | 92.9688 / 99.2188 | +12.5000 | 100.0000 / 100.0000 |
| 34534 / 34534 | 94.5312 / 97.6562 | 91.4062 / 97.1354 | -3.1250 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Timing (lag sign) at 2048 updates: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 47.6562 | 51.8229 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.7500 | 71.9271 | 100.0000 |
| Untrained late encoder | 32 | 53.4375 | 65.1042 | 100.0000 |
| Untrained early encoder | 32 | 52.6563 | 66.9792 | 100.0000 |
| RPB-v7.alt-03 | 32 | 92.1875 | 98.0729 | 100.0000 |
| RPB-v10.alt-01 | 32 | 92.6562 | 98.0208 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 30130 | 78.1250 / 90.8854 | 82.8125 / 90.8854 | +4.6875 | 100.0000 / 100.0000 |
| 31231 / 31231 | 100.0000 / 100.0000 | 97.6562 / 100.0000 | -2.3438 | 100.0000 / 100.0000 |
| 32332 / 32332 | 94.5312 / 100.0000 | 95.3125 / 100.0000 | +0.7812 | 100.0000 / 100.0000 |
| 33433 / 33433 | 91.4062 / 100.0000 | 95.3125 / 100.0000 | +3.9062 | 100.0000 / 100.0000 |
| 34534 / 34534 | 96.8750 / 99.4792 | 92.1875 / 99.2188 | -4.6875 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Timing (lag sign) at 2048 updates: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 48.9062 | 49.4792 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 48.2812 | 52.8125 | 100.0000 |
| Untrained late encoder | 32 | 50.7812 | 52.3958 | 100.0000 |
| Untrained early encoder | 32 | 48.7500 | 52.6042 | 100.0000 |
| RPB-v7.alt-03 | 32 | 87.8125 | 93.8542 | 100.0000 |
| RPB-v10.alt-01 | 32 | 91.7188 | 95.3125 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 30130 | 73.4375 / 82.8125 | 80.4688 / 77.8646 | +7.0312 | 100.0000 / 100.0000 |
| 31231 / 31231 | 98.4375 / 90.6250 | 94.5312 / 99.2188 | -3.9062 | 100.0000 / 100.0000 |
| 32332 / 32332 | 92.1875 / 97.9167 | 95.3125 / 100.0000 | +3.1250 | 100.0000 / 100.0000 |
| 33433 / 33433 | 83.5938 / 98.9583 | 96.8750 / 100.0000 | +13.2812 | 100.0000 / 100.0000 |
| 34534 / 34534 | 91.4062 / 98.9583 | 91.4062 / 99.4792 | +0.0000 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Amplitude transfer at 2048 updates: intact VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 61.4062 | 61.1458 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 54.0625 | 65.4688 | 100.0000 |
| Untrained late encoder | 32 | 99.2188 | 98.2812 | 100.0000 |
| Untrained early encoder | 32 | 99.2188 | 98.6458 | 100.0000 |
| RPB-v7.alt-03 | 32 | 99.8438 | 99.3750 | 100.0000 |
| RPB-v10.alt-01 | 32 | 97.5000 | 97.5521 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 35635 | 100.0000 / 99.7396 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 31231 / 36736 | 100.0000 / 100.0000 | 100.0000 / 100.0000 | +0.0000 | 100.0000 / 100.0000 |
| 32332 / 37837 | 100.0000 / 99.2188 | 91.4062 / 91.9271 | -8.5938 | 100.0000 / 100.0000 |
| 33433 / 38938 | 100.0000 / 99.4792 | 100.0000 / 99.7396 | +0.0000 | 100.0000 / 100.0000 |
| 34534 / 40039 | 99.2188 / 98.4375 | 96.0938 / 96.0938 | -3.1250 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Amplitude transfer at 2048 updates: extra 30% coordinate deletion VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 62.5000 | 46.5104 | 100.0000 |
| Mask only — no encoder | 288 | 50.0000 | 50.0000 | 100.0000 |
| PCA only — no encoder | 32 | 51.4062 | 55.7292 | 100.0000 |
| Untrained late encoder | 32 | 89.2188 | 82.8125 | 100.0000 |
| Untrained early encoder | 32 | 90.9375 | 84.6354 | 100.0000 |
| RPB-v7.alt-03 | 32 | 97.6562 | 94.1667 | 100.0000 |
| RPB-v10.alt-01 | 32 | 92.1875 | 90.0000 | 100.0000 |

Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.

| Timing master / data master | RPB-v7.alt-03 linear / neural % | RPB-v10.alt-01 linear / neural % | Paired linear effect pp | Coverage late / early % |
| --- | ---: | ---: | ---: | ---: |
| 30130 / 35635 | 98.4375 / 94.5312 | 100.0000 / 96.0938 | +1.5625 | 100.0000 / 100.0000 |
| 31231 / 36736 | 99.2188 / 95.5729 | 100.0000 / 97.6562 | +0.7812 | 100.0000 / 100.0000 |
| 32332 / 37837 | 97.6562 / 97.3958 | 78.9062 / 83.0729 | -18.7500 | 100.0000 / 100.0000 |
| 33433 / 38938 | 98.4375 / 91.1458 | 92.1875 / 84.1146 | -6.2500 | 100.0000 / 100.0000 |
| 34534 / 40039 | 94.5312 / 92.1875 | 89.8438 / 89.0625 | -4.6875 | 100.0000 / 100.0000 |

Equal-master means and all five rows are retained. Saved paired effects use the common supported population and a conditional within-master 1,000-replicate/95% source bootstrap. Interval bounds are not averaged into an encoder confidence interval; these budgets are the same continuous trajectories.

## Training and reconstruction

Fixed ordinary whole-patch query standardized MAE uses the same original Q, targets, support and TRAIN scaler across paired architectures and all positive budgets. Inference adds no training context deletion. Complete sampled training Huber/gradient/target traces are separate evidence and remain in the JSON. Point 0 and amplitude reconstruction were not measured.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-03 | 512 | 0.067427298 | 0.068746945 | 13.683443 |
| RPB-v10.alt-01 | 512 | 0.068456936 | 0.069512049 | 14.211626 |
| RPB-v7.alt-03 | 1024 | 0.056892977 | 0.058220361 | 27.221644 |
| RPB-v10.alt-01 | 1024 | 0.059983574 | 0.061028930 | 28.401248 |
| RPB-v7.alt-03 | 2048 | 0.036204511 | 0.037305391 | 54.130685 |
| RPB-v10.alt-01 | 2048 | 0.036075302 | 0.036303958 | 56.940172 |

The training table reports synchronized cumulative update-loop wall time from point 0. Per-interval time is the difference between saved cumulative timers; it is not a second independently trained run.

| Timing master | Updates | RPB-v7.alt-03 TRAIN / VALIDATION MAE | RPB-v10.alt-01 TRAIN / VALIDATION MAE | Cumulative seconds late / early | Interval seconds late / early |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30130 | 512 | 0.074581085 / 0.075675819 | 0.074900193 / 0.075812728 | 13.314956 / 14.274468 | 13.314956 / 14.274468 |
| 31231 | 512 | 0.066458096 / 0.067959707 | 0.070626354 / 0.072530104 | 13.818332 / 14.141764 | 13.818332 / 14.141764 |
| 32332 | 512 | 0.065214569 / 0.068119142 | 0.065735879 / 0.067375543 | 13.683733 / 14.221758 | 13.683733 / 14.221758 |
| 33433 | 512 | 0.063854233 / 0.063252754 | 0.066225168 / 0.064950573 | 13.729818 / 14.168932 | 13.729818 / 14.168932 |
| 34534 | 512 | 0.067028506 / 0.068727305 | 0.064797088 / 0.066891297 | 13.870376 / 14.251206 | 13.870376 / 14.251206 |
| 30130 | 1024 | 0.058589172 / 0.058100192 | 0.065691660 / 0.063439420 | 26.921651 / 28.445762 | 13.606695 / 14.171294 |
| 31231 | 1024 | 0.052339340 / 0.053926753 | 0.050956468 / 0.052378723 | 27.489880 / 28.512991 | 13.671548 / 14.371227 |
| 32332 | 1024 | 0.068211174 / 0.070035895 | 0.068683318 / 0.070677974 | 27.327950 / 28.348028 | 13.644217 / 14.126270 |
| 33433 | 1024 | 0.054693549 / 0.055319339 | 0.061972337 / 0.063618456 | 27.026619 / 28.185891 | 13.296801 / 14.016959 |
| 34534 | 1024 | 0.050631651 / 0.053719625 | 0.052614087 / 0.055030080 | 27.342119 / 28.513567 | 13.471743 / 14.262360 |
| 30130 | 2048 | 0.042440911 / 0.044015136 | 0.037948565 / 0.037626884 | 53.670780 / 56.764808 | 26.749129 / 28.319046 |
| 31231 | 2048 | 0.035140617 / 0.035540850 | 0.037549917 / 0.036058777 | 54.279683 / 57.231649 | 26.789803 / 28.718658 |
| 32332 | 2048 | 0.040702429 / 0.041674677 | 0.030871212 / 0.030836905 | 54.353330 / 57.185413 | 27.025380 / 28.837385 |
| 33433 | 2048 | 0.033053269 / 0.033872207 | 0.044984276 / 0.047039322 | 53.855027 / 56.598548 | 26.828408 / 28.412657 |
| 34534 | 2048 | 0.029685331 / 0.031424087 | 0.029022538 / 0.029957904 | 54.494604 / 56.920444 | 27.152484 / 28.406877 |

Synchronized update-loop wall times include the CUDA training loop and its CPU loss/gradient trace capture. They exclude head fitting, full-feature extraction, reconstruction queries, checkpoint/state writing and byte verification. They are descriptive single-run costs. The separate stages below include CPU transfer, parent verification, file hashing and I/O; pure CUDA kernel time is unmeasured.

| Mixed wall-time scope | Total seconds | Mean per paired cohort seconds |
| --- | ---: | ---: |
| `generation_and_observation_io_seconds` | 0.634099 | 0.126820 |
| `binding_and_checkpoint_io_seconds` | 174.026161 | 34.805232 |
| `CUDA_query_transfer_verification_io_seconds` | 308.146828 | 61.629366 |
| `CUDA_native_transfer_verification_seconds` | 158.199944 | 31.639989 |
| `CPU_baseline_preparation_io_seconds` | 0.938781 | 0.187756 |
| `CPU_head_bootstrap_io_seconds` | 77.742561 | 15.548512 |

## Evidence and limits

Actual supported budgets: 270 pipelines / 540 heads (planned 270/540). All five paired cohorts retain 40 checkpoints, both initial controls, exact live CPU named parameters/buffers/scaler/AdamW states and full trace prefixes. 180 full native export callbacks and 60 query-writer calls produced 240 patch-query banks. Earlier checkpoint/companion bytes were verified after 2,048 before each required quality surface was exported once. Repeated serving parity was tested only with artificial CUDA admission. No extra decoder-only calibration occurred.

Independent audit PASS: 141,317,815 checks, 1,675 CPU archive decodes, 548.286287 seconds. It independently replays saved maps/logits/support/query reductions and source effects; it runs no model, CUDA forward, autodiff, optimizer, head fit or PCA/SVD fit. Ordinary CUDA checkpoint bodies remain byte-bound, with semantics established by source/engineering admission and CPU companions. Original sampler/Torch streams are source/admission-bound, not GPU reenacted.

No adaptive best-point selection, rescue, automatic promotion, TEST or stress occurs. Read the entire paired timing curve, worst-cohort scores, coverage, reconstruction and final amplitude tradeoffs together. Nested known-VALIDATION points do not provide independent generalization confirmation. Strong initial amplitude controls limit training-credit claims. This report does not replace historical v7 weights or prove a causal internal mechanism.

[Frozen prospective card](../../evaluation/cards/early_mixer_learning_curve_v1.md). [Durable full summary](../../../doc/results/early_mixer_learning_curve_v1.json).

Metadata/source bindings:

- `admission_log_sha256`: `a7ab4c5e65da69d8a7f4b5029c55e6d00050e3f363dcfdb675a41b6ae7acf268`
- `admission_sha256`: `4557dd303210f4a80a180db940b16c31441cd1961a0e8576043cb81b70cca83d`
- `audit_reader`: `/embedding/output/runs/rpb-early-mixer-learning-curve/audit-tools/independent-early-mixer-curve-20261009-v3-released-FDv5b0/validate_early_mixer_learning_curve.py`
- `audit_sha256`: `4bcd70194986a0207abcd2e721b6e5bac5cfa62c6b22bf47106d86a09858ebbe`
- `audit_validation`: `/embedding/output/runs/rpb-early-mixer-learning-curve/audit-tools/run-FDv5b0-v3/validation.json`
- `capsule`: `/embedding/output/runs/rpb-early-mixer-learning-curve/early-mixer-learning-curve-FDv5b0`
- `card_sha256`: `869d1a113ed8b6493295632555d69700a279f34e85b01754459761abcc7c4a98`
- `inventory_sha256`: `b153296974bb15d03d181c455f67a79b486d980631ebc2484e5c6ffe5b345d85`
- `reader_sha256`: `6eec9d5f2b73a3e70b907a6d3df3d10ee509f11850e201b8d640f927b7b242e9`
- `renderer_sha256`: `5d6df1a0495eae7b40ef64ee7ecb0bc4d9c430ea3e3402fa718ccc5ff4cc07cd`
- `source_fingerprint`: `7bfda3c2d56716dc7d40c8835fb9f64f1df117969cd4603de3bfa846eb9632f8`
