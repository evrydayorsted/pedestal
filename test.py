# %%
from testPedestal import *

# %%
#test pkl download on all shots and make contour plot
a = Shot("allShots", "pkl")
a.contourPlot(1)

# %%
# test pkl download on single shot, make a contour plot, plot subdata
b = Shot(48339, "pkl")
b.contourPlot(4)
plt.plot(b.W_ped, b.Beta_ped)
plt.xlabel("W_ped")
plt.ylabel("Beta_ped")
plt.show()

# %%
# test client download and make animation
c = Shot(48339, "client")
c.makeAnimation("te")

# %%
# test fit
c.fit(presetTimes = [0.173,0.456])


