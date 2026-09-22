import unittest
from types import SimpleNamespace

from policies.red.baselines import TargetPrior
from policies.red.contracts import Position
from policies.red.tracks import InitialCatalogueTrackFusion


class RedTrackFusionTests(unittest.TestCase):
    def test_detect_info_adds_legally_observed_objective(self) -> None:
        fusion = InitialCatalogueTrackFusion((
            TargetPrior(51, 9400, Position(120.0, 25.0, 0.0), value=10.0),
        ))
        observation = {
            "self": {
                "detectInfo": {
                    168: SimpleNamespace(
                        entity_id=168,
                        entity_type=9500,
                        lla=SimpleNamespace(x=121.0, y=26.0, z=0.0),
                    )
                }
            }
        }

        self.assertTrue(fusion.ingest(observation))
        targets = {item.entity_id: item for item in fusion.targets}
        self.assertEqual(targets[168].entity_type, 9500)
        self.assertEqual(targets[168].position, Position(121.0, 26.0, 0.0))
        self.assertEqual(targets[168].value, 1.0)

    def test_detect_info_ignores_non_objective_track(self) -> None:
        fusion = InitialCatalogueTrackFusion(())
        observation = {
            "self": {
                "detectInfo": {
                    999: SimpleNamespace(
                        entity_id=999,
                        entity_type=24000,
                        lla=SimpleNamespace(x=121.0, y=26.0, z=0.0),
                    )
                }
            }
        }

        self.assertFalse(fusion.ingest(observation))
        self.assertEqual(fusion.targets, ())


if __name__ == "__main__":
    unittest.main()
