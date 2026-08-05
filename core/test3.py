from envengine.sdk.Util import UtilsPy, Vector3D

ecf = UtilsPy.CoordinateHelper.llaToEcef_py(Vector3D(118.416801, 28.225431, 10000))
nue = UtilsPy.CoordinateHelper.ecefToNuePosition_py(ecf, 118.416801, 28.225431)
print(nue.x(), nue.y(), nue.z())
