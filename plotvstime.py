from numpy import array, linspace
import matplotlib.pyplot as plt
#import pyuda
import numpy
import scipy
import pickle


shot = 48136


#        pkldata = {'Shot': shot, 'Times': times, 'W_ped': W_ped, 'Beta_ped': beta_ped,
#                   'W_ped_psin_te': W_ped_psin_te,'W_ped_psin_ne': W_ped_psin_ne,'W_ped_psin_pe': W_ped_psin_pe,
#                   'H_ped_psin_te': H_ped_psin_te,'H_ped_psin_ne': H_ped_psin_ne,'H_ped_psin_pe': H_ped_psin_pe,
#                   'W_ped_radius_te': W_ped_radius_te,'W_ped_radius_ne': W_ped_radius_ne,'W_ped_radius_pe': W_ped_radius_pe,
#                   'H_ped_radius_te': H_ped_radius_te,'H_ped_radius_ne': H_ped_radius_ne,'H_ped_radius_pe': H_ped_radius_pe
#                   'Aratio': Aratio, 'elong': elong, 'delta': delta}

filename = 'output/MAST-U_pedestal_'+str(shot)+'.pkl'
infile = open(filename, 'rb')
pkldata = pickle.load(infile)
infile.close()

       # Plot pedestal radial locations vs. time

#        plt.plot(te_ped_location.time.data, te_ped_location.data, ".-", label="temperature location")
#        plt.plot(ne_ped_location.time.data, ne_ped_location.data, ".-", label="density location")
#        plt.xlabel("Time (s)")
#        plt.ylabel("Pedestal radial position (m)")
#        plt.legend()
#        plt.grid()
#        plt.tight_layout()
#        plt.show()
    
plt.plot(pkldata['Times'], pkldata['W_ped'], label="Width")
plt.plot(pkldata['Times'], pkldata['Beta_ped'], label="Beta")

#plt.plot(pkldata['Times'], pkldata['W_ped_psin_te'], label="Width_psin_te")
#plt.plot(pkldata['Times'], pkldata['H_ped_psin_te'], label="Teped_psin")
#plt.plot(pkldata['Times'], pkldata['W_ped_psin_ne'], label="Width_psin_ne")
#plt.plot(pkldata['Times'], pkldata['H_ped_psin_ne'], label="neped_psin")
#plt.plot(pkldata['Times'], pkldata['W_ped_psin_pe'], label="Width_psin_pe")
#plt.plot(pkldata['Times'], pkldata['H_ped_psin_pe'], label="peped_psin")

#plt.plot(pkldata['Times'], pkldata['Aratio'], label="Aratio")
#plt.plot(pkldata['Times'], pkldata['elong'], label="elong")
#plt.plot(pkldata['Times'], pkldata['delta'], label="delta")



plt.xlabel("Time (s)")
plt.ylabel("Pedestal quantities")
plt.legend()
plt.tight_layout()
plt.show()

#plt.plot(W_ped, beta_ped, label="Beta")
#plt.plot(W_ped_psin_te, H_ped_psin_te, label="Beta")
#plt.xlabel("W_ped")
#plt.ylabel("Beta_ped")
#plt.tight_layout()
#plt.show()


