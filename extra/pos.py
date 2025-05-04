import SPiiPlusPython as sp
hc = sp.OpenCommEthernetTCP("192.168.0.40", 701)
FPOS = sp.GetFPosition(
hc, # communication handle
sp.Axis.ACSC_AXIS_0, # axis 0
failure_check=True
)
print(FPOS)