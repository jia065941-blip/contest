# L Response Package Skill

Coordinates fixed 2-4 L packages after a hidden 9500 target is legally
discovered. It passes through each member's local-network maneuver action and
emits package status, stop-commitment, supplement, and authorized-retarget
requests from a caller-supplied `TargetHealthBelief`.

The package reads no environment or simulator module. Easy plans should start
with triples; hard plans should start with four-member packages. One surviving
member falls back to direct local-network control outside this skill.
