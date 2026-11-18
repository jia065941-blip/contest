from copy import deepcopy
from pathlib import Path
import unittest

from envengine.environment.training_env import TrainingEnv
from scenarios import load_scenario
from user_agents.example.common.commander import RedBaselineCommander
from user_agents.example.common.config import RedAgentConfig
from user_agents.example.common.contracts import Position
from user_agents.example.common.coordination import _GroupMailbox
from user_agents.example.common.targets import initial_targets_from_observation
from user_agents.red_checker import check_deployment, check_red_agent


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / 'scenarios/E01.json'
ENTRY = ROOT / 'user_agents/example/coordination/r7_v1_0_strike_packages.py'


class CommunicationMemoryTests(unittest.TestCase):
    def real_group(self, policy_name='r7_strike_packages'):
        env = TrainingEnv(load_scenario(E01), record_logs=False)
        self.addCleanup(env.close)
        env.reset()
        manager = env.agent_manager
        deployment = check_deployment(ENTRY)()
        self.addCleanup(deployment.close)
        manager.apply_deployment(deployment.deploy(manager.get_deployment_observation(seed=42)))
        group = manager.register_user_agent(check_red_agent(ENTRY), seed=42)
        self.addCleanup(group.close)
        env.engine.refresh_red_communications()
        full = env.get_observation()
        reports = [manager.extract_observation_for_agent(full, agent.entity_id)
                   for agent in group.agents]
        self.assertTrue(all(reports))
        groups = {}
        for report in reports:
            members = frozenset({report['entity_id'], *report['self']['commRangeInfo']})
            groups.setdefault(members, []).append(report)
        members, reports = max(groups.items(), key=lambda item: len(item[1]))
        self.assertGreater(len(reports), 1)
        reports.sort(key=lambda report: report['entity_id'])
        initial = group.agents[0].init_observation
        targets = initial_targets_from_observation(initial)
        area = initial['map_area']
        polygon = tuple(Position(*point) for point in (
            (area['xMin'], area['yMin']), (area['xMax'], area['yMin']),
            (area['xMax'], area['yMax']), (area['xMin'], area['yMax']),
        ))
        config = RedAgentConfig(policy_name=policy_name, seed=42)
        registered = {agent.entity_id for agent in group.agents}
        mailbox = _GroupMailbox(members, registered, targets, config, polygon)
        reference = RedBaselineCommander(targets, config.policy_name, config.seed, polygon)
        for entity_id in sorted(members & registered):
            reference.register_platform(entity_id)
        return mailbox, reference, reports

    def test_real_scene_reports_are_released_without_changing_planning(self):
        mailbox, reference, reports = self.real_group()
        step = reports[0]['step']
        reference.begin_step(deepcopy(tuple(reports)))
        for report in reports:
            mailbox.receive(report)
        commander = mailbox.prepare(step)
        self.assertEqual(mailbox._reports, {})
        self.assertEqual(commander.reports, reference.reports)
        self.assertEqual(
            [commander.action_for(report['entity_id'], step) for report in reports],
            [reference.action_for(report['entity_id'], step) for report in reports],
        )
        self.assertEqual(commander.target_by_platform, reference.target_by_platform)
        tasks = dict(commander.target_by_platform)
        guidance = dict(commander.guidance)
        self.assertIs(mailbox.prepare(step), commander)
        self.assertEqual(commander.target_by_platform, tasks)
        self.assertEqual(commander.guidance, guidance)

    def test_real_scene_reports_keep_caller_isolation(self):
        mailbox, reference, reports = self.real_group()
        step = reports[0]['step']
        reference.begin_step(deepcopy(tuple(reports)))
        identifiers = [report['entity_id'] for report in reports]
        for report in reports:
            mailbox.receive(report)
            report.clear()
        commander = mailbox.prepare(step)
        self.assertEqual(commander.reports, reference.reports)
        self.assertEqual(
            [commander.action_for(entity_id, step) for entity_id in identifiers],
            [reference.action_for(entity_id, step) for entity_id in identifiers],
        )
        self.assertEqual(mailbox._reports, {})

    def test_real_scene_nested_changes_do_not_change_optimized_plan(self):
        for policy_name in ('r0_random', 'r7_strike_packages', 'r7_static_search'):
            with self.subTest(policy_name=policy_name):
                mailbox, reference, reports = self.real_group(policy_name)
                step = reports[0]['step']
                reference.begin_step(deepcopy(tuple(reports)))
                for report in reports:
                    mailbox.receive(report)
                    report['self']['pos'].clear()
                    report['self']['health'] = 0
                    report['self']['isVisible'] = False
                    report['self']['commRangeInfo'].clear()
                    for track in report['self']['detectInfo'].values():
                        if track.get('pos') is not None:
                            track['pos'].clear()
                    report['self']['detectInfo'].clear()
                commander = mailbox.prepare(step)
                self.assertEqual(commander.reports, reference.reports)
                self.assertEqual(commander.targets, reference.targets)
                self.assertEqual(
                    [commander.action_for(report['entity_id'], step) for report in reports],
                    [reference.action_for(report['entity_id'], step) for report in reports],
                )

    def test_other_policies_keep_complete_isolated_reports(self):
        mailbox, reference, reports = self.real_group('r1_priority')
        step = reports[0]['step']
        reference.begin_step(deepcopy(tuple(reports)))
        for report in reports:
            mailbox.receive(report)
            self.assertEqual(mailbox._reports[report['entity_id']], report)
            report['self']['commRangeInfo'].clear()
            report['self']['pos'].clear()
        commander = mailbox.prepare(step)
        self.assertEqual(commander.reports, reference.reports)
        self.assertEqual(
            [commander.action_for(report['entity_id'], step) for report in reports],
            [reference.action_for(report['entity_id'], step) for report in reports],
        )


if __name__ == '__main__':
    unittest.main()
