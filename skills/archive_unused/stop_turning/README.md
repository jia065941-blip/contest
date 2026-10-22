# stop_turning

Emit the V13-compatible stop/straight action (`1`) and official engine lateral
command `0.0` for exactly the requested number of simulator steps. This clears
the desired lateral acceleration; it does not stop forward flight.

```python
from stop_turning_skill import StopTurningSkill

skill = StopTurningSkill().start(execution_steps=10)
while skill.remaining_steps:
    command = skill.step()
    actions.append([0, entity_id, command.engine_acc_z, 0.0])
```

`step()` raises `StateError` before `start()` and after the requested duration.
Call `reset()` or `start()` after completion to run it again.
