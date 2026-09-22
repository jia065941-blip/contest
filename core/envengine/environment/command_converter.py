from envengine.agent_manager.actions.aircraft_action import SetDesiredAccZ, MissileLaunchAction, ChangeTargetAction
from envengine.agent_manager.actions.aircraft_action.use_satellite import UseSatelliteAction
from envengine.agent_manager.actions.deploy_action import SetLLA
from envengine.common import Vector3d


class CommandConverter:
    """
    指令转换器
    """
    @staticmethod
    def common_converter(actions):
        """
        将np.array格式的指令转换为结构体指令
        :return: 结构体指令
        """
        actions_converted = []
        for action in actions:
            for action_item in action:
                # 设置Z轴加速度
                if action_item[0] == 0:
                    actions_converted.append(
                        SetDesiredAccZ(
                            executor_id=int(action_item[1]),
                            acc_z=float(action_item[2])
                        ).to_dict()
                    )

                # 发射导弹
                if action_item[0] == 1:
                    actions_converted.append(
                        MissileLaunchAction(
                            executor_id=int(action_item[1]),
                            target=Vector3d(
                                float(action_item[2]),
                                float(action_item[3]),
                                0
                            )
                        ).to_dict()
                    )

                # 改变目标
                if action_item[0] == 2:
                    actions_converted.append(
                        ChangeTargetAction(
                            executor_id=int(action_item[1]),
                            target=Vector3d(
                                float(action_item[2]),
                                float(action_item[3]),
                                0
                            )
                        ).to_dict()
                    )

                # 使用卫星
                if action_item[0] == 3:
                    actions_converted.append(
                        UseSatelliteAction(
                            executor_id=int(action_item[1]),
                        ).to_dict()
                    )

                # 首次 LAUNCH 同帧设置进入场景坐标。TrainingEnv 在送入
                # Engine 前直接执行并过滤该 DEPLOY 命令。
                if action_item[0] == 4:
                    actions_converted.append(
                        SetLLA(
                            executor_id=int(action_item[1]),
                            lla=Vector3d(
                                float(action_item[2]),
                                float(action_item[3]),
                                float(action_item[4]),
                            ),
                        ).to_dict()
                    )

        return actions_converted
