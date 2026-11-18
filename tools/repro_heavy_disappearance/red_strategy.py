from user_agents.example.spatial.runtime import SpatialAttack, SpatialDeployment


class Deployment(SpatialDeployment):
    variant = 'recovery'


class RedAttack(SpatialAttack):
    variant = 'recovery'
