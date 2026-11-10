# Exploratory OR01-COMB probe on the noun hypernym interior

Date: 31 October 2026, completed 1 November 2026.
Method: OR01-COMB on the 17,191-entry local-text snapshot `or01-ret-abs-0.2.0-local-text`.
Gates, fixed in `bindings/oewn-2025/interior-run-gates.json` before any weight was read: tau 0.05, gamma 0.2, rho 3, kappa 1.
Claim level: exploratory_description. Weights are fit coefficients.

Each query is a noun leaf from `bindings/oewn-2025/leaf-queries.json`. The corpus is the encoded interior. The registered exact-search tool was not used. IdyDB stored the same vectors and answered its own cosine nearest neighbor; that query is not this probe.

## Ranked leaves

| Query | Cosine | Path | Seconds | Face then support | Largest weights |
| --- | --- | --- | --- | --- | --- |
| puppy 01325095-n | 0.970 | 1.064 | 0.76 | 14 then 7 | pup 0.395, toy spaniel 0.151, lamb 0.140, piglet 0.116, foal 0.086, young fish 0.073, guide dog 0.072 |
| wolf pup 01324999-n | 0.968 | 1.058 | 0.29 | 8 then 6 | pup 0.402, coyote 0.184, cub 0.142, foal 0.141, lamb 0.079, piglet 0.075 |
| kitty 02124950-n | 0.947 | 1.071 | 97.9 | 21 then 4 | domestic cat 0.428, chat 0.282, wildcat 0.157, calf 0.148 |
| tandem bicycle 02838476-n | 0.958 | 1.063 | 95.3 | 21 then 4 | bicycle 0.581, bench 0.194, two-handed saw 0.164, bicycle race 0.079 |
| burin 02925845-n | 0.950 | 1.068 | 186 | 22 then 5 | chisel 0.281, needle 0.279, engraving 0.224, drill 0.129, graver 0.102 |
| anadama bread 07695913-n | 0.954 | 1.090 | 568 | 23 then 8 | cornbread 0.225, bread 0.163, white bread 0.156, rum 0.141, baba 0.109, Anasazi 0.093, custard apple 0.091, dark bread 0.063 |

All six returned outcome ranked, reason none. Reconstruction cosine sat between 0.947 and 0.970. Path ratio sat between 1.058 and 1.090, under the declared ceiling of 3.

## Leaves that did not finish the tie search

The current solver enumerates subsets of the KKT face after Lawson-Hanson, from the smallest width up to the support size. Faces of width 21–23 finished. Wider faces did not:

| Query | Face | What happened |
| --- | --- | --- |
| jalopy 02927938-n | 27 | Killed after 27833 s at 97.9 million subsets, width 13 |
| ball-peen hammer 02786058-n | 27 | timeout 1200 s |
| barong 02797391-n | 28 | timeout 1200 s |
| crab apple 07755320-n | 24 | timeout 1200 s |

This probe does not close `or-01-retrieval-geometric-combination-run`. That milestone still waits on replacing the normal-equation step and the combinatorial face walk.

Log: ignored `.temp/operations/retrieval/or01-ret-abs-0.2.0-local-text/combine-leaves.log`.
