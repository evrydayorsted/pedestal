from Shot import *
import time
start_time = time.time()
a = Shot(49239, "all")


# Adding new shots
a.fit(savepklforshot=True)
print("--- %s seconds ---" % (time.time() - start_time))
a.contourPlot(1)
