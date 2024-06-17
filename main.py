from Shot import *
import time
import pandas
start_time = time.time()

newShots = pandas.read_csv("MU03_shotlist_cleaned.csv")
counter = 1
for i in newShots["Shot Number"]:
    a = Shot(i, "all")
    # Adding new shots
    a.fit(savepklforshot=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(str(176-counter)+"left to go")
    counter += 1
a.contourPlot(1)
