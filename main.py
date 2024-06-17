from Shot import *

a = Shot("allShots", "pkl")

a.contourPlot(8, numPix=60, plotName="allShotsDeltaVsKappaAratioColor30px10min", cbarMin=1.4, cbarMax=1.9, countType="aratio",numMin=10)
# a.contourPlot(1, numPix=30, plotName="allShotsBetaVsDeltaAratioColor30px20min", cbarMin=1.52, cbarMax=1.56, countType="aratio",numMin=20)
# a.contourPlot(1, numPix=40, plotName="allShotsBetaVsDeltaAratioColor40px10min", cbarMin=1.52, cbarMax=1.56, countType="aratio",numMin=10)
# a.contourPlot(1, numPix=40, plotName="allShotsBetaVsDeltaAratioColor40px20min", cbarMin=1.52, cbarMax=1.56, countType="aratio",numMin=20)
