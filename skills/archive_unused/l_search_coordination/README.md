# L Search Coordination Skill

Coordinates legal 4x8 search-cell ownership, discovery fusion, member-loss
reassignment requests, and delayed satellite requests. It never changes search
waypoints and never reads simulator/global state. Member maneuver actions pass
through the caller-provided local-network baseline.
