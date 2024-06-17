from Shot import *
import time
start_time = time.time()

shots = []

for i in shots:
    a = Shot(shots[i], "all")
    # Adding new shots
    a.fit(savepklforshot=True)
print("--- %s seconds ---" % (time.time() - start_time))
a.contourPlot(1)
