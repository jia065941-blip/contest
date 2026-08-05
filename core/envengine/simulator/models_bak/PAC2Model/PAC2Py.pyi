from typing import List

from UtilsPy import Vector3D, State_py


class Missile:
    """
    Missile model (ModelDevelop)
    """

    def __init__(self) -> None: ...

    def __del__(self) -> None: ...


    def Save(flag: bool) -> None:



    # -------------------------
    # Init
    # -------------------------
    def Init(self,step:float, m_lla: Vector3D) -> None:



    # -------------------------
    # Launch
    # -------------------------
    def Launch(theta_f_d:float, psi_f_d:float) -> None:



    def SetTargetEcf(self, targetPosEcf: Vector3D, targetVelEcf: Vector3D)-> None:



    def SetTargetLLA(self, targetPosLLa: Vector3D, targetVelEcf: Vector3D)-> None:



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