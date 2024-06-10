"""
"""


# Imports

import pyuda
import numpy
import matplotlib.pyplot as plt

client = pyuda.Client()

shot = 44661

betaN = client.get('epm/output/globalParameters/betaN',shot)
betaN_data = numpy.array(betaN.data)
betaN_time = numpy.array(betaN.time.data)

plt.plot(betaN_time,betaN_data)

plt.show()










