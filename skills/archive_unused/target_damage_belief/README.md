# Target Damage Belief Skill

Maintains lower/expected/upper target-health estimates using only data supplied
by the caller. The package imports no simulator or environment module.

Legal current inputs are initial public target health, configured hidden-target
health priors, own missile assignment/lifecycle evidence, delivery heuristics,
and confirmed destruction. `HitReceipt` is supported as a future legal API but
is not currently emitted by the official environment.

Commitments affect projected health only. Heuristic delivery affects the
expected and lower-health bounds but never claims exact damage. A confirmed hit
receipt replaces the heuristic estimate for the same missile.
