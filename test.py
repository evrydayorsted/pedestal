import unittest
from pedestal import *

class TestPedestal(unittest.TestCase):

    def testPklDownload(self):
        allShots = Shot("allShots", "pkl", folder = "outputWithPlasmaCurrent")
        W_ped = np.array([0.09493417, 0.20379311, 0.05938031, ..., 0.05535727, 0.05810597,
       0.39816288])
        self.assertEqual(calculation.get_sum(), 10, 'The sum is wrong.')

if __name__ == '__main__':
    unittest.main()