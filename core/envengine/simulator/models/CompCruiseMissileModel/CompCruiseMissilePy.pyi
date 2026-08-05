from typing import List

from UtilsPy import Vector3D, State_py


class Missile:
    """
    Missile model (ModelDevelop)
    """

    def __init__(self) -> None: ...

    def __del__(self) -> None: ...
    # -------------------------
    # Init
    # -------------------------
    def Init(self,step:float, m_lla: Vector3D,maxLoad:float) -> None:


    def Save(flag: bool) -> None:


    # -------------------------
    # Launch
    # -------------------------
    def Launch(t_lla: Vector3D) -> None:
        """
        Launch missile with pitch and yaw commands (deg)
        """
    def SetDesiredHeight(self, height: float)-> None:

    def SetDesiredAccZ(self, accZ: float)-> None:
    # -------------------------
    # Target setting
    # -------------------------
    def SetTargetEcf(
            self,
            targetPosEcf: Vector3D,
            targetVelEcf: Vector3D,
    ) -> None:
        """
        Set target state in ECEF
        """

    # -------------------------
    # Update
    # -------------------------
    def update(self) -> float:
        """
        Step missile dynamics forward by one time step
        Returns: step time or internal status flag
        """

    # -------------------------
    # State access
    # -------------------------
    def getState(self) -> State_py:
        """
        Return current missile state
        """

    def SetDesiredSpeed(self, speed:float)->None:

