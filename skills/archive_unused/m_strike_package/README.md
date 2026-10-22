# M Strike Package Skill

Pure status/request coordination for immutable 2-6 M/21001 packages assigned
to one public 9400 or 9600 target within a ten-step release window. The selected
local network remains the sole maneuver authority: `motion_overrides` is always
empty. The skill emits conservative stop, staged supplement, and authorized
retarget requests from a caller-supplied `TargetHealthBelief`.
