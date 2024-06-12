from Shot import *


a = Shot("allShots", "pkl")

a.contourPlot(1, savefigure=False)
plt.savefig("plots/belowMeanDelta-OneStdev.png")
plt.show()