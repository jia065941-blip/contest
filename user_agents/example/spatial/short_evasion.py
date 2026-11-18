from math import isfinite, sqrt


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class ShortThreatEvasion:
    def __init__(self):
        self.samples = {}
        self.end_step = None
        self.next_allowed = 0
        self.bursts = 0
        self.handled = {}

    def command(self, observation):
        step = observation['step']
        if self.end_step is not None:
            if step >= self.end_step:
                self.end_step = None
                self.next_allowed = step + 8
                return 0.0
            return None
        own = observation['self']
        if own.get('type') not in (21000, 21001):
            return None
        point = own['pos']
        risks = []
        for raw_id, track in (own.get('detectInfo') or {}).items():
            if field(track, 'entity_type') != 24000:
                continue
            key = int(field(track, 'entity_id', raw_id))
            timestamp, position = field(track, 'detected_step'), field(track, 'pos')
            if timestamp is None or position is None:
                continue
            relative = tuple(field(position, axis) - point[axis] for axis in ('x', 'y', 'z'))
            if not isfinite(timestamp) or not all(isfinite(v) for v in relative):
                raise ValueError('自爆虫记录需要有限步号和米制位置')
            previous = self.samples.get(key)
            if previous is not None and timestamp <= previous[1]:
                continue
            self.samples[key] = (step, timestamp, relative)
            if previous is None or not 0 < step - previous[0] <= 3:
                continue
            dt = step - previous[0]
            velocity = tuple((a-b)/dt for a,b in zip(relative, previous[2]))
            speed_squared = sum(v*v for v in velocity)
            if speed_squared <= 1e-9:
                continue
            encounter = -sum(a*b for a,b in zip(relative, velocity))/speed_squared
            separation = sqrt(sum(v*v for v in relative))
            miss = sqrt(sum((a+encounter*b)**2 for a,b in zip(relative, velocity)))
            if 0 < encounter <= 25 and separation <= 12000 and miss <= 1000 and step - self.handled.get(key, -1000) >= 15:
                risks.append((encounter, key))
        if not risks or step < self.next_allowed or self.bursts >= 8:
            return None
        _, key = min(risks)
        self.handled[key] = step
        self.end_step = step + 2
        self.bursts += 1
        return 0.12 if self.bursts % 2 else -0.12
