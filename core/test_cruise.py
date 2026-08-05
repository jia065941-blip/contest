import datetime
import time

from envengine.sdk.Util import UtilsPy
# from envengine.simulator.models.CompCruiseMissileModel.CompCruiseMissileHPy import Missile
from envengine.simulator.models.CompCruiseMissileModel.CompCruiseMissilePy import Missile


dt = datetime.datetime(2000, 1, 1, 0, 0, 0)

c = Missile()
missileLLA = UtilsPy.Vector3D(117.100999, 26.46319, 10000)
targetLLA = UtilsPy.Vector3D(122.459791, 23.167392, 0)
c.Save(False)
c.SetDesiredSpeed(1000)
c.Init(0.05, missileLLA, 20)

c.SetDesiredHeight(8000)

c.Launch(targetLLA)
ret = 0
for i in range(500 * 200):
    start = time.perf_counter()
    ret = c.update()
    state = c.getState()

    lla2 = UtilsPy.CoordinateHelper.ecefToLla_py(state.posEcf())
    print(lla2.x(), lla2.y(), lla2.z())

    end = time.perf_counter()
    elapsed_us = (end - start) * 1e6
    # print(f"耗时: {elapsed_us:10.3f} 微秒")

    if ret > 0:
        print(ret)
        break
    dt += datetime.timedelta(seconds=0.05)
