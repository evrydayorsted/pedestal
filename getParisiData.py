"""
"""


# Imports
from numpy import array, linspace
import matplotlib.pyplot as plt
import pyuda
import numpy
import scipy
import pickle

def getClientData(shot):
	print(shot)
	client = pyuda.Client()
	te   = client.get('/ayc/t_e',shot)
	print("te downloaded")
	dte  = client.get('/ayc/dt_e',shot)
	print("dte downloaded")
	ne   = client.get('/ayc/n_e',shot)
	print("ne downloaded")
	dne  = client.get('/ayc/dn_e',shot)
	print("dne downloaded")
	r    = client.get('/ayc/r',shot)
	print("r downloaded")
	psinprof  = client.get('epm/output/radialprofiles/normalizedpoloidalflux',shot)

	times_ayc = te.time.data
	print("All Data Downloaded")
	return (te,dte,ne,dne,r, psinprof,times_ayc)


W_ped = pkldata['W_ped']
Beta_ped = pkldata['Beta_ped']
