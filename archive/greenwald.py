from pedestal import *
import pyuda
import numpy as np
import time
def calc_ne_bar(shot):

    # From Dave Taylor's IDL code ne_bar_dt
    # Ne_bar=ane_density/(4*(SQRT(r_out^2-r_tan^2)-SQRT(r_in^2-r_tan^2)))
    
    r_tan = 356.805*1.0E-3
    
    equil_present = False

    if shot is not None:
        equil = Shot(shot, "pkl")
        equil_present = True
    
  
        
    if equil_present is False:
        raise Exception('calc_ne_bar: calculation cannot proceed without equilibrium data.')
    client=pyuda.Client()
    
    density_data = client.get('ane/density', shot)
    rmidin    = client.get('epm/output/separatrixGeometry/rmidplaneIn',shot)
    rmidout   = client.get('epm/output/separatrixGeometry/rmidplaneOut',shot)
    aminor    = client.get('epm/output/separatrixGeometry/minorRadius',shot)

    output_data = np.zeros(len(equil.times))
    
    for i in np.arange(len(output_data)):
        if (equil.times[i] > np.min(density_data.time.data)) and (equil.times[i] < np.max(density_data.time.data)):
            tmp1 = np.interp(equil.times[i], rmidout.time.data, rmidout.data)**2 - r_tan**2
            tmp2 = np.interp(equil.times[i], rmidin.time.data, rmidin.data)**2 - r_tan**2
            if tmp1 < 0: tmp1 = 0
            if tmp2 < 0: tmp2 = 0
            output_data[i] = np.interp(equil.times[i], density_data.time.data, density_data.data) / (4.0 * (np.sqrt(tmp1) - np.sqrt(tmp2)))

    # Since all of the data is already here, calculate the Greenwald density and fraction
    ngw = 1.0E20 * equil.Ip * 1.0E-6 / (np.pi * np.interp(equil.times, aminor.time.data, aminor.data)**2)
    
    greenwald_fraction = output_data/ngw

    output =  {'data': output_data, 
               't': equil.times,
               'greenwald_density': ngw,
               'greenwald_fraction': greenwald_fraction}
    
    return output
failedShotNums = []
if __name__ == '__main__':
    for i in allShotNums:
        try:
            start_time = time.time()

            a = calc_ne_bar(i)
            pkldata = a
            filename = 'outputWithBeamPower3/outputWithTagData/greenwald_'+str(i)+'.pkl'
            outfile = open(filename, 'wb')
            pickle.dump(pkldata,outfile)
            outfile.close()
            print('outputWithBeamPower3/outputWithTagData/greenwald_'+str(i)+'.pkl saved')
            print("--- %s seconds ---" % (time.time() - start_time))

        except Exception as error:
            print(error)
            failedShotNums += [i]

print(failedShotNums)
print(len(failedShotNums))