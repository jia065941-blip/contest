import datetime
import random
import time

from envengine.sdk.Util.UtilsPy import CoordinateHelper, Vector3D

from envengine.sdk.Util import UtilsPy
from envengine.simulator.models.CompCruiseMissileModel import CompCruiseMissilePy

from envengine.simulator.models.PAC2Model import PAC2Py

step = 0.01
dt = datetime.datetime(2000, 1, 1, 0, 0, 0)
pac2 = PAC2Py.Missile()
cruise = CompCruiseMissilePy.Missile()
# pac2.Save(True)
# cruise.Save(False)

missileLLA = UtilsPy.Vector3D(118.416801, 28.225431, 10001.165784196928)
missileEcf = CoordinateHelper.llaToEcef_py(missileLLA)
targetLLA = UtilsPy.Vector3D(121.609607, 27.644557, 0)

targetPosNue = CoordinateHelper.ecefToNuePosition_py(missileEcf, targetLLA.x(), targetLLA.y())
theta_f = CoordinateHelper.getTheta_py(targetPosNue) * 57.3
psi_f = CoordinateHelper.getPsi_py(targetPosNue) * 57.3
# 800 ~ 1200
cruise.Init(step, missileLLA, 20)
cruise.SetDesiredSpeed(1000)
cruise.Launch(UtilsPy.Vector3D(122.083302, 27.652843, 0))


pac2.Init(step, targetLLA)
state = cruise.getState()
pac2.SetTargetEcf(state.posEcf(), Vector3D(0, 0, 0))

pac2.Launch(30, psi_f)
ret = 0
for i in range(int(500 / step)):
    ret1 = cruise.update()
    num = random.randint(1, 200)

    # ret1 = 0
    if ret1 > 0:
        print("cruise", ret1)
        break
    state = cruise.getState()
    if i % 20 == 0:
        pac2.SetTargetEcf(state.posEcf(), state.velEcf())

    ret2 = pac2.update()

    state2 = pac2.getState()

    lla2 = UtilsPy.CoordinateHelper.ecefToLla_py(state2.posEcf())
    print(lla2.x(), lla2.y(), lla2.z())

    if ret2 > 0:
        print(dt)
        print("pac2", ret2)
        break


    dt += datetime.timedelta(seconds=step)

#  雷达 探测到100KM以内的 红方弹 -> 把信息传递给指控-> 计算偏角 -> 选一个弹发射，并记录哪个ID的拦截弹拦截哪个ID的进攻弹 -> 指控给拦截弹指令 更新目标
