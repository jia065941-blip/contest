from ..common.baselines import distance_km


class StrikeCoordination:
    # 与搜索任务相同的45步宽限；过期仅表示预算不再可靠，不表示单位死亡。
    REPORT_GRACE_STEPS = 45

    def __init__(self, deployment, targets):
        self.assignments = {int(k): v for k, v in deployment['assignments'].items()}
        self.targets = {t.entity_id: t for t in targets}
        self.health = {v['target_id']: v['health'] for v in deployment['damage_budget']}
        self.remaining = {v['target_id']: v['nominal_committed_damage'] for v in deployment['damage_budget']}
        self.accounted_losses = set()
        self.redirected = set()
        self.redirects = []
        self.strike_reports = {}
        self.expired_commitments = {}
        self.speed_km_step = deployment['speed_km_step']
        self.navigation_rejections = {}
        self.search_seen = set()
        self.search_claims = []
        arrivals = [(v['arrival_step'], self.targets[v['target_id']].value * self.damage(v['kind'], v['target_id']) / self.health[v['target_id']])
                    for v in self.assignments.values() if v['role'] == 'strike']
        # 每个候选主攻波次前保留20步接近时间。
        starts = sorted({max(0, time-20) for time, _ in arrivals})
        self.satellite_start = max(starts, key=lambda start: (
            sum(weight for time, weight in arrivals if start <= time < start+80), -start))
        self.satellite_step = None

    def damage(self, kind, target_id):
        return 15 if kind == 'H' else 5

    def restore_reports(self, reports):
        for key, target, step, distance in reports:
            previous = self.strike_reports.get(key)
            if previous is None or step > previous[1]:
                self.strike_reports[key] = (target, step, distance)

    def record_report(self, key, unit, target, aim, step):
        if unit.alive and unit.kind in ('H', 'M') and target in self.targets:
            self.strike_reports[key] = (target, step, distance_km(unit.position, aim))

    def refresh_commitments(self, reports, target_by_platform, last_aim, step):
        for key, unit in reports.items():
            target = target_by_platform.get(key)
            if target in self.targets:
                self.record_report(key, unit, target,
                                   last_aim.get(key, self.targets[target].position), step)
        expired = {
            key: target for key, (target, seen, distance) in self.strike_reports.items()
            if step > seen + self.REPORT_GRACE_STEPS and distance > 2
            and key not in self.accounted_losses
        }
        # 差量恢复允许重新联络的单位重新计入，避免重复扣减和永久误判。
        for key, target in self.expired_commitments.items():
            self.remaining[target] += self.damage(self.assignments[key]['kind'], target)
        for key, target in expired.items():
            self.remaining[target] -= self.damage(self.assignments[key]['kind'], target)
        self.expired_commitments = expired

    def repair_losses(self, reports, target_by_platform, last_aim, local_tracks, step, velocities, stages):
        """按当前可信预算补打；失联、确认死亡和末端消失分别处理。"""
        self.refresh_commitments(reports, target_by_platform, last_aim, step)
        for key, unit in reports.items():
            target_id = target_by_platform.get(key)
            if unit.kind not in ('H', 'M') or unit.alive or target_id not in self.targets or key in self.accounted_losses:
                continue
            self.accounted_losses.add(key)
            expired_target = self.expired_commitments.pop(key, None)
            if expired_target is not None:
                self.remaining[expired_target] += self.damage(unit.kind, expired_target)
            aim = last_aim.get(key, self.targets[target_id].position)
            if distance_km(unit.position, aim) > 2:
                self.remaining[target_id] -= self.damage(unit.kind, target_id)
        affected = {target for target in self.targets if self.remaining[target] < self.health[target]}
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
                    track = local_tracks[key].get(target_id)
                    # 补打只使用新近探测；公开初值及过期记录不代表目标仍在该处。
                    if stages[key] != 2 or track is None or not 0 <= step - track[0] <= 1:
                        continue
                    old_aim = last_aim.get(key, self.targets[old].position)
                    new_aim = track[1]
                    old_distance, new_distance = distance_km(unit.position, old_aim), distance_km(unit.position, new_aim)
                    if old_distance < 5 or new_distance > old_distance*1.3:
                        continue
                    vx, vy = velocities[key]
                    # 原生巡航中只接受45度以内的改派，避免大角度横向追赶。
                    forward = ((new_aim.x - unit.position.x) * vx
                               + (new_aim.y - unit.position.y) * vy)
                    distance_squared = ((new_aim.x - unit.position.x) ** 2
                                        + (new_aim.y - unit.position.y) ** 2)
                    within_turn = forward > 0 and 2 * forward ** 2 >= distance_squared * (vx ** 2 + vy ** 2)
                    arrival = step + new_distance / self.speed_km_step[unit.kind]
                    if not within_turn or arrival > 1800:
                        self.navigation_rejections.setdefault((key, target_id), {
                            'step': step, 'platform_id': key, 'target_id': target_id,
                            'forward_dot': forward, 'estimated_arrival': arrival,
                            'reason': 'turn_exceeds_45_degrees' if not within_turn else 'arrival_after_mission_window'})
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
                self.record_report(key, unit, target_id, aim, step)
                self.redirects.append({'step': step, 'platform_id': key, 'from_target': old, 'to_target': target_id,
                                       'reason': 'insufficient_current_strike_commitment',
                                       'position': [unit.position.x, unit.position.y],
                                       'velocity': list(velocities[key]), 'aim': [aim.x, aim.y],
                                       'public_stage': stages[key],
                                       'target_detected_step': local_tracks[key][target_id][0],
                                       'estimated_arrival': step + distance_km(unit.position, aim) / self.speed_km_step[unit.kind],
                                       'expired_unconfirmed': sorted(k for k, target in self.expired_commitments.items()
                                                                     if target == target_id)})
        return commands

    def diagnostics(self):
        return {'satellite_planned_start': self.satellite_start, 'satellite_actual_step': self.satellite_step,
                'own_loss_reassignments': self.redirects, 'search_observed_dark_ids': sorted(self.search_seen),
                'search_claims': self.search_claims,
                'expired_unconfirmed_commitments': dict(self.expired_commitments),
                'remaining_commitment_note': 'nominal allocation excluding confirmed losses and expired en-route reports; NOT enemy remaining health'}
