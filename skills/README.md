# Concrete Skills

Each child directory is an independently installable concrete Skill that depends
on the shared root `skill-generation` Runtime. Keep model-specific dependencies,
trusted Backends, adapters, bundles and tests inside that Skill directory.

Current packages:

- `local_avoidance`: network-agnostic loader and contract for a caller-supplied
  local-avoidance runtime and trained checkpoint.
- `h_single_maneuver`: H-only CPA rule composed from the three timed maneuver
  Skills, with PlanSpec-v2/B0 hybrid-versus-local-V13 comparison modes.
- `three_missile_joint`: variable 2-6 H strike packages layered sparsely over
  V13; includes an exact-three compatibility wrapper.
- `target_damage_belief`: legal-observation target-health intervals, delivery
  evidence, in-flight commitment, and optional confirmed hit receipts.
- `l_search_coordination`: 4x8 search-cell ownership, discovery quorum,
  reassignment requests, and delayed satellite requests for search L units.
- `l_response_package`: 2-4 L hidden-target response packages with damage
  belief, stop-commitment, supplement, and authorized-retarget requests.
- `l_satellite_strike_packages`: one globally deduplicated satellite request,
  legal hidden-nest track admission, and configurable deterministic L packages.
- `staggered_deployment`: deterministic continuous-area deployment with L in
  the front band and H/M in a separated rear band.
- `hm_public_strike_packages`: expected-damage H/M allocation over public
  9400/9600 targets with legal same-ID coordinate refreshes.
- `target_damage_probability`: evidence-only delivered and projected target
  destruction probabilities with replayable immutable snapshots.
- `dynamic_retarget`: bounded communication-domain H/M retarget decisions for
  flexible missiles after a delivered destruction threshold is met.
- `terminal_guidance_override`: owner-local M terminal straightening layered
  after the external avoidance policy with fresh-track and threat gates.
- `reserve_echelon`: group-local H/L reserve package state machine for early
  commitment-loss supplementation and deterministic fallback.
- `simple_upper_orchestrator`: official release adapter combining staggered
  deployment, L/HM planning and a caller-supplied local avoidance network.

Legacy and currently unused skills are retained under `archive_unused/` and are
not part of the active simple-upper stack.
- `mission_replan`: event-driven PlanSpec overlay that stages L response
  packages, supplements on delivery timeout, and cancels satisfied targets.
- `m_strike_package`: immutable 2-6 M public-target packages with pair/follow-up
  roles, damage belief, and stop/supplement/authorized-retarget requests.
- `turn_left`: emit action 0 / `-20G` for a requested number of steps.
- `turn_right`: emit action 2 / `+20G` for a requested number of steps.
- `stop_turning`: emit action 1 / `0G` for a requested number of steps; forward
  flight continues.

Unified functions, DTO inputs/outputs, authority boundaries, runtime modes, and
legal-information rules are documented in
[`SKILL_INTERFACES.md`](SKILL_INTERFACES.md).

New Skills should use their own `pyproject.toml`, import shared Runtime contracts,
register only their trusted Backend IDs and avoid adding domain dependencies to
the root framework.
