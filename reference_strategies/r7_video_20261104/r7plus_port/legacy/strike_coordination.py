from .baselines import distance_km


class StrikeCoordination:
    def __init__(self, deployment, targets):
        self.assignments = {int(k): v for k, v in deployment['assignments'].items()}
        self.targets = {t.entity_id: t for t in targets}
        self.health = {v['target_id']: v['health'] for v in deployment['damage_budget']}
        self.remaining = {v['target_id']: v['nominal_committed_damage'] for v in deployment['damage_budget']}
        self.accounted_losses = set()
        self.redirected = set()
        self.redirects = []
        self.search_seen = set()
        self.search_claims = []
        arrivals = [(v['arrival_seconds'], self.targets[v['target_id']].value * self.damage(v['kind'], v['target_id']) / self.health[v['target_id']])
                    for v in self.assignments.values() if v['role'] == 'strike']
        # Reserve 20 seconds for the approach before each candidate main wave.
        starts = sorted({max(0, time-20) for time, _ in arrivals})
        self.satellite_start = max(starts, key=lambda start: (
            sum(weight for time, weight in arrivals if start <= time <= start+180), -start))
        self.satellite_step = None

    def damage(self, kind, target_id):
        return (20 if kind == 'H' else 5) * (.8 if self.targets[target_id].entity_type == 9400 else .6)

    def repair_losses(self, reports, target_by_platform, last_aim, local_tracks, step):
        """Replace confirmed en-route own losses, never infer enemy destruction.

        Disappearance near the aim point is ambiguous and is not subtracted.
        Existing spent/terminal commitments are retained as estimates, not kills.
        A donor must leave its old objective's original nominal budget intact.
        """
        affected = set()
        for key, unit in reports.items():
            target_id = target_by_platform.get(key)
            if unit.kind not in ('H', 'M') or unit.alive or target_id not in self.targets or key in self.accounted_losses:
                continue
            self.accounted_losses.add(key)
            aim = last_aim.get(key, self.targets[target_id].position)
            if distance_km(unit.position, aim) > 2:
                self.remaining[target_id] -= self.damage(unit.kind, target_id)
                affected.add(target_id)
        commands = {}
        for target_id in sorted(affected, key=lambda k: (-self.targets[k].value, k)):
            while self.remaining[target_id] < self.health[target_id]:
                candidates = []
                for key, unit in reports.items():
                    old = target_by_platform.get(key)
                    if not unit.alive or unit.kind not in ('H', 'M') or old not in self.targets or old == target_id or key in self.redirected:
                        continue
                    if self.remaining[old]-self.damage(unit.kind, old) < self.health[old]:
                        continue
                    old_aim = last_aim.get(key, self.targets[old].position)
                    new_aim = local_tracks.get(key, {}).get(target_id, (None, self.targets[target_id].position))[1]
                    old_distance, new_distance = distance_km(unit.position, old_aim), distance_km(unit.position, new_aim)
                    if old_distance < 5 or new_distance > old_distance*1.3:
                        continue
                    candidates.append((new_distance, key, old, new_aim))
                if not candidates:
                    break
                _, key, old, aim = min(candidates)
                unit = reports[key]
                self.remaining[old] -= self.damage(unit.kind, old)
                self.remaining[target_id] += self.damage(unit.kind, target_id)
                self.redirected.add(key)
                commands[key] = (target_id, aim)
                self.redirects.append({'step': step, 'platform_id': key, 'from_target': old, 'to_target': target_id,
                                       'reason': 'own_loss_more_than_2km_before_last_aim'})
        return commands

    def diagnostics(self):
        return {'satellite_planned_start': self.satellite_start, 'satellite_actual_step': self.satellite_step,
                'own_loss_reassignments': self.redirects, 'search_observed_dark_ids': sorted(self.search_seen),
                'search_claims': self.search_claims,
                'remaining_commitment_note': 'nominal allocation after confirmed en-route losses; NOT enemy remaining health'}
