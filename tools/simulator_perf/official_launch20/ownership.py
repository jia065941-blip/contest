"""Remove one redundant report copy for the audited frozen Astra actors.

The SDK already gives each member its own isolated snapshot. PlanAgent,
PlanGroup, ReceivedTrackFusion and V13 only read that snapshot. Reports may
therefore retain the same snapshot within this member's actor/group pipeline.
Task dictionaries remain copied because the planner actively modifies them.
"""

def receive(self, observation, task):
    import base_agents
    pid = int(observation['entity_id'])
    if pid not in self.members:
        raise ValueError('Cross-group report')
    self.reports[pid] = (observation, base_agents.copy.deepcopy(task))


def install():
    import agents
    agents.PlanGroup.receive = receive
