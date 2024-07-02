from Shot import *
import time
import pandas
start_time = time.time()

shots = [
47885,47886,47888]#,47889,47890,47891,47893,47894,47918,47955,47956,47958,47959,47961,47962,47963,47964,47979,47980,47982,47985,47989,47990,47991,47996,47997,47998,47999,48000,48002,48003,48004,48005,48006,48008, 48009, 48010, 48012, 48057, 48058, 48060, 48061, 48062, 48064, 48065, 48066, 48068, 48069, 48070, 48071, 48072, 48073, 48074, 48079, 48080, 48081, 48082, 48083, 48087, 48088, 48089, 48093, 48094,48103,48104,48107,48108,48109,48110,48111,48112,48113,48114,48115,48116,48117,48118,48119,48120,48121,48122,48123,48124,48125,48126,48127,48129,48130,48131,48132,48133,48134,48135,48136,48137,48151,48155,48156,48157,48158,48159,48160,48164,48168,48172,48173,48174,48175,48176,48177,48178,48180,48181,48183,48186,48187,48188,48189,48193,48194,48196,48198,48200,48219,48221,48223,48233,48235,48251,48252,48255,48256,48257,48258,48259,48260,48261,48263,48265,48267,48268,48269,48270,48271,48272,48273,48275,48276,48278,48279,48280,48281,48284,48285,48286,48287,48288,48291,48292,48293,48295,48297,48298,48299,48302,48303,48304,48305,48309,48310,48311,48312,48313,48314,48315,48316,48326,48330,48332,48333,48334,48336,48337,48338,48339,48340,48341,48342,48343,48344,48345,48347,48348,48353,48354,48359,48361,48363,48366,48367,48368,48369,48370,48558,48559,48560,48561,48579,48580,48594,48595,48596,48597,48598,48599,48602,48603,48604,48605,48606,48609,48611,48614,48615,48616,48617,48618,48619,48620,48622,48623,48630,48631,48632,48634,48636,48638,48639,48640,48641,48642,48643,48646,48647,48648,48649,48651,48652,48653,48654,48655,48656,48657,48658,48666,48668,48669,48670,48671,48672,48710,48711,48712,48714,48715,48716,48717,48718,48721,48722,48723,48725,48726,48735,48738,48740,48743,48745,48749,48750,48752,48755,48758,48759,48760,48761,48762,48763,48764,48765,48766,48767,48768,48769,48772,48777,48778,48779,48780,48788,48789,48791,48797,48798,48799,48800,48801,48802,48803,48804,48805,48806,48807,48808,48809,48811,48812,48813,48816,48817,48818,48819,48820,48821,48822,48823,48824,48825,48826,48827,48828,48829,48830,48832,48834,48835,48836,48840,48841,48842,48844,48845,48846,48847,48849,48850,48851,48853,48863,48864,48866,48867,48868,48869,48870,48871,48872,48873,48874,48879,48880,48882,48883,48884,48885,48886,48888,48889,48890,48892,48893,48894,48895,48896,48898,48899,48900,48901,48902,48903,48904,48906,48907,48908,48909,48910,48911,48912,48913,48915,48916,48917,48918,48919,48920,48921,48925,48926,48927,48928,48929,48930,48931,48932,48933,48934,48935,48936,49033,49034,49035,49036,49037,49038,49039,49040,49042,49045,49046,49047,49048,49049,49050,49051,49052,49054,49055,49056,49057,49058,49059,49060,49061,49062,49063,49066,49069,49070,49071,49072,49073,49074,49075,49076,49077,49078,49080,49081,49084,49091,49093,49094,49095,49099,49101,49102,49103,49104,49105,49106,49107,49108,49109,49110,49111,49112,49113,49117,49118,49119,49120,49121,49122,49123,49124,49125,49126,49127,49128,49130,49131,49134,49135,49136,49137,49138,49139,49140,49141,49142,49143,49145,49146,49147,49148,49149,49150,49151,49152,49154,49157,49159,49162,49163,49164,49166,49167,49168,49169,49170,49171,49172,49173,49174,49175,49177,49178,49179,49180,49181,49182,49183,49184,49186,49187,49188,49189,49190,49191,49192,49194,49195,49196,49197,49198,49200,49204,49205,49206,49208,49209,49210,49211,49212,49213,49214,49216,49217,49218,49219,49220
#]
# 47956,47982, 48071, 48156, 48291, 49063 , 49069 failed Ip
failedShots = []
newShots = pandas.read_csv("MU03_shotlist_cleaned.csv")
counter = 1
totalNumShots = len(shots)+len(newShots["Shot Number"])
for i in shots:
    try:
        a = Shot(i, "both")
        # Adding new shots
        a.fit(savePklForShot=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(str(totalNumShots-counter)+"left to go")
        counter += 1
    except Exception as error:
        counter +=1
        failedShots += [i]
        print(error)
        print(i, " FAILED --------------------------------------")
        
for i in newShots["Shot Number"]:
    try:
        a = Shot(i, "both")
        # Adding new shots
        a.fit(savePklForShot=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(str(totalNumShots-counter)+"left to go")
        counter += 1
    except Exception as error:
        counter += 1
        failedShots += [i]
        print(error)
        print(i, " FAILED -------------------------")

print("failedShots", failedShots)
