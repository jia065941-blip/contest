# Mission Replan Skill

Pure event-driven mission arbitration layered over a static PlanSpec. The first
implementation stages L hidden-response packages: after all required legal
discoveries and the original 20-step delay, v2 grants whole packages covering
at least half each hidden-target allocation, waits for delivery/BDA, supplements
on calibrated timeouts, and cancels unreleased packages only on confirmed death
or high-confidence delivered overkill. Projected commitments cannot permanently
cancel a package.

It reads no simulator/global truth and emits no maneuver commands. PlanSpec
remains the initial-plan owner; R9 remains an untouched official baseline.

The PlanSpec adapter adds a constrained cleanup-retarget overlay for packages
whose current target is confirmed/high-confidence complete. It chooses the
nearest compatible legal, live, undercommitted target, rate-limits package
movement, emits one normal PlanSpec goal decision, and atomically transfers the
shared damage commitment after the command is accepted.
