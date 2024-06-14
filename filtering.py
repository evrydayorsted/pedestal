from Shot import *

a = Shot("allShots", "pkl")

# a.contourPlot(9, fitHMode=False,savefigure=True, plotName= "allShotsDeltavsAratioLimited2pm0.02")
plt.scatter(x=a.Aratio, y=a.delta, c=a.times, cmap="rainbow", s= 0.6, alpha=0.6) 
plt.xlim(1, 3)
plt.xlabel(r"Aspect Ratio")
plt.ylabel(r"$\delta$")
plt.ylim(0,.75)
plt.colorbar(label="Time in shot", orientation="horizontal") 
plt.savefig("allShotsAratioVsDeltaTimeColored.png")
plt.show()
#plt.savefig("plots/elongAboveMean+OneStdev.png")