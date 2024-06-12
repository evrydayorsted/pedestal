from Shot import *

a = Shot("allShots", "pkl")
fig, ax = plt.subplots(1, 1) 
ax.hist(a.Aratio, bins = 200, range=(1.3,1.9)) 
  
# Set title 
ax.set_title("Aspect Ratio Histogram") 
  
# adding labels 
ax.set_xlabel('Aratio') 
ax.set_ylabel('counts') 
plt.savefig("plots/aRatioHistogram")
plt.show()