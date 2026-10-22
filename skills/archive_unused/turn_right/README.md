# turn_right

Emit the V13-compatible right action (`2`) and official engine lateral command
`+1.0` (`+20G`) for exactly the requested number of simulator steps.

```python
from turn_right_skill import TurnRightSkill

skill = TurnRightSkill().start(execution_steps=10)
while skill.remaining_steps:
    command = skill.step()
    actions.append([0, entity_id, command.engine_acc_z, 0.0])
```

`step()` raises `StateError` before `start()` and after the requested duration.
Call `reset()` or `start()` after completion to run it again.
