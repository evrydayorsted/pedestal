from Shot import *
import time
import pandas
# start_time = time.time()

# newShots = pandas.read_csv("MU03_shotlist_cleaned.csv")
# counter = 1
# for i in newShots["Shot Number"]:
#     if i>49379:
#         a = Shot(i, "all")
#         # Adding new shots
#         a.fit(savepklforshot=True)
#         print("--- %s seconds ---" % (time.time() - start_time))
#         print(str(79-counter)+"left to go")
#         counter += 1

paperShots = [45261, 45270, 45272]

for i in paperShots:
    a = Shot(i, "all")
    a.fit(savepklforshot=True)
    