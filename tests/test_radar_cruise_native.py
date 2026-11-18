from pathlib import Path
import unittest

from simulator import Simulator


class RadarCruiseNativeTests(unittest.TestCase):
    def test_actual_cruise_flights_are_detected_within_twenty_kilometers(self):
        root = Path(__file__).resolve().parents[1]
        simulator = Simulator({
            'agent': root / 'user_agents/example/coordination/r7_v4_0_search_recovery.py',
            'scenario': root / 'scenarios/E06.json', 'seed': 7,
            'blue_mode': root / 'blue_defense/immediate/b0_fixed_ratio_random.py',
            'render_mode': 'none',
            'logging': {'enabled': False, 'example_evaluation': False},
        })
        covered = set()
        try:
            simulator._initialize()
            factory = simulator._env.engine.simulator_factory
            for radar in factory.get_simulators_by_type(44000):
                execute, handle = radar.execute_detection, radar.handel_detect_info
                delivered = {}

                def capture(records, handle=handle, delivered=delivered):
                    delivered.update(records)
                    return handle(records)

                def checked(radar=radar, execute=execute, delivered=delivered):
                    delivered.clear()
                    expected = set()
                    origin = radar.entity_ext.entity.posEcf
                    for target in factory.get_simulators_by_side(0):
                        entity = target.entity_ext.entity
                        squared = sum((getattr(entity.posEcf, axis) - getattr(origin, axis)) ** 2
                                      for axis in ('x', 'y', 'z'))
                        if (entity.survivePoints > 0 and entity.isVisible
                                and entity.stage >= 2 and squared <= 20000 ** 2):
                            expected.add(entity.id)
                    execute()
                    self.assertEqual(set(delivered), expected)
                    for identifier in delivered:
                        model = factory.get_simulator_by_id(identifier)
                        if model.entity_ext.entity.stage == 2:
                            self.assertEqual(int(model.model.getState().stage()), 3)
                            covered.add(identifier)

                radar.handel_detect_info = capture
                radar.execute_detection = checked
            # 保持真实飞行状态、策略与命令处理，仅审计探测输出。
            for _ in range(450):
                _, _, done, _ = simulator._step()
                if done:
                    break
            self.assertTrue(covered, '原生飞行未覆盖公开阶段2探测')
        finally:
            simulator._close(None)


if __name__ == '__main__':
    unittest.main()
