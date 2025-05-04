import SPiiPlusPython as sp

hc = sp.OpenCommEthernetTCP("192.168.0.40", 701)
sp.Disable(hc, 0, sp.SYNCHRONOUS, True)
#sp.Enable(hc, 0, sp.SYNCHRONOUS, True)