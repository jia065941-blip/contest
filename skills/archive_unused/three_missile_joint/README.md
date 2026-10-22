# H Strike Package Skill

Sparse coordination for one fixed package of 2-6 H missiles. All members must
share target ID and a bounded release window. A single H always remains on the
direct V13 path; packages larger than six must be split by a higher layer. V13
remains the baseline for every missile; this skill materializes final actions
through `turn_left`, `stop_turning`, and `turn_right`.

The initial rule changes at most one action only for a predicted peer collision
below 150 metres within three seconds. Threat ownership and ETA spread are
recorded but do not override V13. Groups are immutable and never refill after a
member is lost. Any invalid or incomplete group falls back transactionally to
independent V13 decisions. `ThreeMissileJointSkill` remains as an exact-three
compatibility wrapper over `HStrikePackageSkill`.

The skill also assigns deterministic `decoy`, `primary`, and `backup` roles,
publishes role action preferences and a strike-package status, and can emit a
`RetargetRequest` when a caller-supplied health belief confidently marks the
current target destroyed. It never executes semantic retargeting itself.

No simulator or environment module is imported. Every observation, belief, and
authorization is supplied explicitly by the caller.
