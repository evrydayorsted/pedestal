# %%
from testPedestal import *
import time
# %%
#test pkl download on all shots and make contour plot
a = Shot("allShots", "pkl")
a.contourPlot(1)
print("\n\n")
# %%
# test pkl download on single shot, make a contour plot, plot subdata
b = Shot(48339, "all")
b.contourPlot(4)
plt.plot(b.W_ped, b.Beta_ped)
plt.xlabel("W_ped")
plt.ylabel("Beta_ped")
plt.show()
print("\n\n")
# %%
# test client download and make animation
b.makeAnimation("te")
print("\n\n")
# %%
# test fit
b.fit(presetTimes = [0.400], plotvsradius = True)
print("\n\n")

c = Shot(48340, "client")
c.fit(presetTimes = [0.406], plotvsradius = True)
print("Tests ran successfully")



