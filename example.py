import os
import numpy as np

import PySide6
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from pyappac2d.appac import panel2d

filepath = os.path.abspath(os.path.dirname(__file__))

surfaceFiles = ["onr-dep/mainVec0.dat", "onr-dep/nacelleVec0.dat"]
surfaces = []
for i in range(len(surfaceFiles)):
    with open(os.path.join(filepath, "airfoils", surfaceFiles[i]), encoding="utf-8") as f:
        surfaces.append(np.loadtxt((line.replace(',',' ') for line in f)))

alphaDeg = 10.

Cp, xc, foils, wakes = panel2d(surfaces, alphaDeg, 1.)

app = pg.mkQApp()
win = pg.GraphicsLayoutWidget(show=True, title="pyappac2d")
p1 = win.addPlot(row=0, col=0, title="Pressure Distribution")
p1.invertY()
for i in range(len(Cp)):
    p1.plot(xc[i],Cp[i])
p2 = win.addPlot(row=1, col=0, title="Airfoil System")
p2.setAspectLocked(True)
for i in range(len(surfaces)):
    p2.plot(surfaces[i][:,0], surfaces[i][:,1])

if __name__ == '__main__':
    pg.exec()