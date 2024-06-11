# %%
from Shot import *
import time
import numpy as np
# %%
#test pkl download on all shots and make contour plot
a = Shot("allShots", "pkl")
a.contourPlot(1, savefigure= False, showfigure = False)
print("\n\n")
# %%
# test pkl download on single shot, make a contour plot, plot subdata
b = Shot(48340, "all")
b.contourPlot(4, savefigure= False, showfigure = False)
plt.plot(b.W_ped, b.Beta_ped, ".", label="data")
plt.plot(b.W_ped[45], b.Beta_ped[45], "r.", markersize= 10, label="406 ms")
x = np.linspace(0., 0.3, 100)
y = (x/0.2)**(1/0.91)
plt.plot(x,y, label = r"$\delta = 0.2 \beta^{0.91}$ (wide)")
plt.ylim(0, 0.35)
z = x/0.13
plt.plot(x,z, label = r"$\delta= 0.13 \beta$ (narrow)")

plt.xlabel("W_ped")
plt.ylabel("Beta_ped")
plt.title("Shot 48340")
plt.legend()
#plt.savefig("plots/parisi_48340_scatter.png")
#plt.show()
print("\n\n")
# %%
# test client download and make animation
b.makeAnimation("te", saveanim= False)
print("\n\n")
# %%
# test fit
b.fit(presetTimes = [0.406], plotvsradius = True, savefigure= False, showfigure = False)
print("\n\n")

#c = Shot(48339, "client")
#c.fit(presetTimes = [0.400], plotvsradius = True, savefigure= False, plotfigure = False)
print("Tests ran successfully")



