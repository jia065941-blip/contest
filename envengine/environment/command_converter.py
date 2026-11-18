"""Convert the current red action protocol without interpreting hidden state."""

import numpy as np

from envengine.action_validation import IllegalActionError
from envengine.agent_manager.actions.aircraft_action import SetDesiredAccZ, ChangeTargetAction
from envengine.agent_manager.actions.aircraft_action.use_satellite import UseSatelliteAction
from envengine.common import Vector3d


class CommandConverter:
    @staticmethod
    def common_converter(actions) -> list[dict]:
        """Accept target coordinates (1), satellite (2), and maneuver (3).

        Deployment starts every red unit. An explicit launch action is invalid.
        Target IDs stay inside the strategy; the public action supplies a point.
        """
        converted = []
        for raw_action in actions:
            try:
                rows = np.asarray(raw_action, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise IllegalActionError("动作必须是数值数组") from exc
            if rows.shape in ((0,), (0, 4)):
                continue
            if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
                raise IllegalActionError("动作必须是有限数值 N x 4 数组")
            for action_type, executor_id, value, other in rows:
                if not float(executor_id).is_integer() or executor_id < 0:
                    raise IllegalActionError("执行实体编号必须是非负整数")
                executor_id = int(executor_id)
                if action_type == 1:
                    command = ChangeTargetAction(
                        executor_id=executor_id,
                        intent="waypoint",
                        target=Vector3d(float(value), float(other), 0.0),
                    )
                elif action_type == 2:
                    if value != 0 or other != 0:
                        raise IllegalActionError("天眼动作未使用的参数位置必须填 0")
                    command = UseSatelliteAction(executor_id=executor_id)
                elif action_type == 3:
                    if other != 0:
                        raise IllegalActionError("机动动作未使用的参数位置必须填 0")
                    command = SetDesiredAccZ(executor_id=executor_id, acc_z=float(value))
                else:
                    raise IllegalActionError(f"不支持动作类型 {action_type}；有效类型为 1、2、3")
                converted.append(command.to_dict())
        return converted
