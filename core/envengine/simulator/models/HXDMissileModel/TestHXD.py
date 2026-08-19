import datetime
import time
import UtilsPy
from UtilsPy import CoordinateHelper
import HXDMissilePy

# 高、中性能弹

step = 0.05
dt = datetime.datetime(2000, 1, 1, 0, 0, 0)
targetLLA  = UtilsPy.Vector3D (113, 40, 0)

M1 = HXDMissilePy.BatchMissile.getInstance()

M1.SetMissileCount(1)

MLLA = UtilsPy.Vector3D (120, 40, 10)
M1.Init(0, step, 20, MLLA)
M1.SetDesiredHeight(0,30000)

M1.Save(0,True)


M1.Launch(0, targetLLA)
for i in range(int(1000 / step)):
    M1.Update(i* step)
    ret1 = M1.getRet(0)
    if ret1>0:
        print("m1: ",ret1)
        break

    dt += datetime.timedelta(seconds=step)