print("Importing libraries...")

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
from   matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
import pickle
from archive.contourplot import *
import copy
import numpy as np
import scipy
from scipy.optimize import curve_fit
import time
import pandas


try:
    from pedinf.models import mtanh # type: ignore
except:
    print("pedinf connection failed")

try:
    import pyuda # type: ignore
except:
    print("pyuda connection failed")

print("Libraries imported.")
print("\n")

class Shot:
    def __init__(self, shotNum, datatype, folder="output20240715"):
        """Initializes shot object

        Args:
            shotNum (int or str):  Shot # or 'allShots'
            datatype (str): 'pkl', 'client', or 'both'
        """
        # Save shotNum as a string so it can be used in filenames
        self.shotNum = str(shotNum)

        # Initialize data pull instructions
        self.pkl = False
        self.client = False

        # Define functions to pull data
        def pklDownload(self):
            """Pulls data from pkl file 'folder/MAST_U_pedestal_{#}.pkl' or 'folder/MAST_U_pedestal_allShots.pkl'"""
            print("Downloading pkl data...", self.shotNum)
            try:
                #download pkl
                filename = folder +'/MAST-U_pedestal_'+self.shotNum+'.pkl'
                infile = open(filename, 'rb')
                pkldata = pickle.load(infile)
                infile.close()
                #read off values
                try:
                    self.shot = pkldata['shot']
                    self.times = pkldata['times']
                    self.aratio = pkldata['aratio']
                    self.shotIndexed = pkldata['shotIndexed']
                except:
                    #deprecated variable names
                    self.shot = pkldata['Shot']
                    self.times = pkldata['Times']
                    self.aratio = pkldata['Aratio']
                    self.shotIndexed = pkldata['ShotNum']

                self.W_ped = pkldata['W_ped']
                self.Beta_ped = pkldata['Beta_ped']
                self.W_ped_psin_te = pkldata['W_ped_psin_te']
                self.W_ped_psin_ne = pkldata['W_ped_psin_ne']
                self.W_ped_psin_pe = pkldata['W_ped_psin_pe']
                self.H_ped_psin_te = pkldata['H_ped_psin_te']
                self.H_ped_psin_ne = pkldata['H_ped_psin_ne']
                self.H_ped_psin_pe = pkldata['H_ped_psin_pe']
                self.W_ped_radius_te = pkldata['W_ped_radius_te']
                self.W_ped_radius_ne = pkldata['W_ped_radius_ne']
                self.W_ped_radius_pe = pkldata['W_ped_radius_pe']
                self.H_ped_radius_te = pkldata['H_ped_radius_te']
                self.H_ped_radius_ne = pkldata['H_ped_radius_ne']
                self.H_ped_radius_pe = pkldata['H_ped_radius_pe']
                self.elong  = pkldata['elong']
                self.delta = pkldata['delta']
                self.Ip = np.array(pkldata["Ip"])
                self.IpMax = np.array(pkldata['IpMax'])
                try:
                    self.NBI = pkldata["NBI"]
                    self.ssNBI = pkldata["ssNBI"]
                    self.swNBI = pkldata["swNBI"]
                    self.betaN = pkldata["betaN"]

                except:
                    print("No NBI data found")
                self.nullity = pkldata["nullity"]
                self.beamPower = pkldata["beamPower"]
                self.divertor = pkldata["divertor"]
                self.whichBeams = pkldata["whichBeams"]
                self.pkl = True
                try:
                    self.elmTimesAc = pkldata["elmTimesAc"]
                    self.elmTimesNorm = pkldata["elmTimesNorm"]
                    self.greenwaldFraction = pkldata["greenwaldFraction"]
                    self.elmPercent = pkldata["elmPercent"]
                except:
                    print("No elm data found")
                print("Pkl data loaded")
                print("\n")


            except Exception as error:
                print(error)
                print("Pkl data procurement failed")
                print("\n")

        def clientDownload(self):
            '''Function to pull data from client. Should only be used for a single shot.'''
            print("Getting data from client for " +self.shotNum+" ...")
            try:
                #define parameters for pulling from uda
                client = pyuda.Client()
                shot = self.shotNum

                # Pull fit data from client
                group = "/apf/core/mtanh/lfs/"
                self.te_ped_location = client.get(group + "t_e/pedestal_location", shot)
                self.te_ped_height   = client.get(group + "t_e/pedestal_height", shot)
                self.te_ped_width    = client.get(group + "t_e/pedestal_width", shot)
                self.te_ped_top_grad = client.get(group + "t_e/pedestal_top_gradient", shot)
                self.te_background   = client.get(group + "t_e/background_level", shot)

                self.times_apf       = self.te_ped_location.time.data

                self.ne_ped_location = client.get(group + "n_e/pedestal_location", shot)
                self.ne_ped_height   = client.get(group + "n_e/pedestal_height", shot)
                self.ne_ped_width    = client.get(group + "n_e/pedestal_width", shot)
                self.ne_ped_top_grad = client.get(group + "n_e/pedestal_top_gradient", shot)
                self.ne_background   = client.get(group + "n_e/background_level", shot)

                self.pe_ped_location = client.get(group + "p_e/pedestal_location", shot)
                self.pe_ped_height   = client.get(group + "p_e/pedestal_height", shot)
                self.pe_ped_width    = client.get(group + "p_e/pedestal_width", shot)
                self.pe_ped_top_grad = client.get(group + "p_e/pedestal_top_gradient", shot)
                self.pe_background   = client.get(group + "p_e/background_level", shot)

                print("Done downloading pedestal parameters")

                # EPM: EFIT++
                self.Ip        = client.get('epm/output/globalParameters/plasmacurrent',shot)
                self.times_epm = self.Ip.time.data
                #toroidal magnetic field (at axis)
                self.Btor      = client.get('epm/output/globalParameters/bphirmag',shot)
                self.betaN      = client.get('epm/output/globalParameters/betan',shot)
                # storedEnergy (joules)
                self.plasmaEnergy      = client.get('epm/output/globalParameters/plasmaEnergy',shot)
                print("5/20 efit parameters loaded", end="\r")

                self.rmaxis    = client.get('epm/output/globalParameters/magneticAxis/R',shot)
                self.zmaxis    = client.get('epm/output/globalParameters/magneticAxis/Z',shot)
                self.rbdy      = client.get('epm/output/separatrixGeometry/rboundary',shot)
                self.zbdy      = client.get('epm/output/separatrixGeometry/zboundary',shot)
                self.rmidin    = client.get('epm/output/separatrixGeometry/rmidplaneIn',shot)
                print("10/20 efit parameters loaded", end="\r")

                self.rmidout   = client.get('epm/output/separatrixGeometry/rmidplaneOut',shot)
                self.aminor    = client.get('epm/output/separatrixGeometry/minorRadius',shot)
                self.kappa     = client.get('epm/output/separatrixGeometry/elongation',shot)
                self.deltaup   = client.get('epm/output/separatrixGeometry/upperTriangularity',shot)
                self.deltalow  = client.get('epm/output/separatrixGeometry/lowerTriangularity',shot)
                #self.pmaxis    = client.get('epm/output/globalParameters/psiAxis',shot)
                #self.psibdy    = client.get('epm/output/globalParameters/psiBoundary',shot)
                print("15/20 efit parameters loaded", end="\r")

                self.rprof     = client.get('epm/output/radialprofiles/R',shot)
                print("16/20 efit parameters loaded", end="\r")

                self.psinprof  = client.get('epm/output/radialprofiles/normalizedpoloidalflux',shot)
                print("17/20 efit parameters loaded", end="\r")

                self.r_2D      = client.get('epm/output/profiles2D/R',shot)
                print("18/20 efit parameters loaded", end="\r")

                self.z_2D      = client.get('epm/output/profiles2D/Z',shot)
                print("19/20 efit parameters loaded", end="\r")

                self.psin_2D   = client.get('epm/output/profiles2D/psinorm',shot)
                print("20/20 efit parameters loaded", end="\r")

                Ip = client.get('epm/output/globalParameters/plasmacurrent',self.shotNum)
                self.Ip = np.nan_to_num(Ip.data)
                
                self.IpTime = Ip.time.data
                #self.psi_2D    = client.get('epm/output/profiles2D/poloidalflux',shot)
                print("Done downloading efit parameters", end="\n")
                
                try:
                    #in megawatts
                    self.total_NBI_power = client.get('anb/sum/power', shot)
                except:
                    self.total_NBI_power = None
                    print("Total NBI fail")
                try:
                    #in megawatts
                    self.ss_NBI_power = client.get('xnb/ss/beampower', shot)
                except:
                    self.ss_NBI_power = None
                    print("SS NBI fail")
                try:
                    #in megawatts
                    self.sw_NBI_power = client.get('xnb/sw/beampower', shot)
                except:
                    self.sw_NBI_power = None
                    print("SW NBI fail")
                print("Done downloading NBI power", end="\n")


                # thomson data
                self.te   = client.get('/ayc/t_e',self.shotNum)
                print("1/6 ayc parameters loaded", end="\r")
                self.dte  = client.get('/ayc/dt_e',self.shotNum)
                print("2/6 ayc parameters loaded", end="\r")
                self.ne   = client.get('/ayc/n_e',self.shotNum)
                print("3/6 ayc parameters loaded", end="\r")
                self.dne  = client.get('/ayc/dn_e',self.shotNum)
                print("4/6 ayc parameters loaded", end="\r")
                self.r    = client.get('/ayc/r',self.shotNum)
                print("5/6 ayc parameters loaded", end="\r")
                self.psinprof  = client.get('epm/output/radialprofiles/normalizedpoloidalflux',self.shotNum)
                print("6/6 ayc parameters loaded", end="\r")
                self.rprof     = client.get('epm/output/radialprofiles/R',self.shotNum)
                print("Done downloading ayc parameters", end="\n")
                self.times_ayc = self.te.time.data


                self.client = True
                print("All data downloaded from client")
                print("\n")

            except Exception as error:
                print(error)
                print("Client connection failed.")
    
        # Run the appropriate data pulls
        if datatype == "pkl":
            pklDownload(self)
        elif datatype == "client":
            clientDownload(self)
        elif datatype == "both":
            pklDownload(self)
            clientDownload(self)
        else:
            raise Exception("datatype must be 'pkl,' 'client,' or 'both'")
        
    def __str__(self):
        '''Define string representation'''
        return f"{self.shotNum}" 
    
    def fit(self, printTimes= False, plotVsTime = False, printQuantities = False,
	    plotVsRadius = False, plotVsPsiN = False, savePklForShot = False,
	    presetTimes= [], saveFigure = False, showFigure = False):
        """_summary_

        Args:
            printTimes (bool, optional): Prints the time slices that are processed. Defaults to False.
            plotVsTime (bool, optional): Plots radial pedestal locations vs time. Defaults to False.
            printQuantities (bool, optional): Prints W_ped and Beta_ped for each equilibria. Defaults to False.
            plotVsRadius (bool, optional): Plots thompson data and fit against radius. Defaults to False.
            plotVsPsiN (bool, optional): Plots fit against psin. Defaults to False.
            savePklForShot (bool, optional): Saves a pkl file with shot parameters to outputWithPlasmaCurrent/MAST-U_pedestal_{shot#}.pkl. Defaults to False.
            presetTimes (list, optional): List of time slices to process. Defaults to all time slices within a shot.
            saveFigure (bool, optional): Saves plot to "plots/{shot#}_{timeSliceInMs}_plotVs{PsiN or radius}.png". Defaults to False.
            showFigure (bool, optional): Displays figure in matplotlib window. Defaults to False.
       
        Adapted from pedestal_fit_parameters Jack Berkery 2023"""
        if not self.client:
            raise Exception("Must have client data to run fit")
        

        te_ped_location = self.te_ped_location
        te_ped_height   = self.te_ped_height
        te_ped_width    = self.te_ped_width
        te_ped_top_grad = self.te_ped_top_grad
        te_background   = self.te_background
        times_ayc       = self.times_ayc
        times_apf       = self.times_apf
        times_epm       = self.times_epm
        ne_ped_location = self.ne_ped_location
        ne_ped_height   = self.ne_ped_height
        ne_ped_width    = self.ne_ped_width
        ne_ped_top_grad = self.ne_ped_top_grad
        ne_background   = self.ne_background
        pe_ped_location = self.pe_ped_location
        pe_ped_height   = self.pe_ped_height
        pe_ped_width    = self.pe_ped_width
        pe_ped_top_grad = self.pe_ped_top_grad
        pe_background   = self.pe_background

        # group the temperature and density parameters together so it's easy
        # to pass them into the model:
        te_parameters = np.array([
            te_ped_location.data,
            te_ped_height.data,
            te_ped_width.data,
            te_ped_top_grad.data,
            te_background.data
        ])

        ne_parameters = np.array([
            ne_ped_location.data,
            ne_ped_height.data,
            ne_ped_width.data,
            ne_ped_top_grad.data,
            ne_background.data
        ])

        pe_parameters = np.array([
            pe_ped_location.data,
            pe_ped_height.data,
            pe_ped_width.data,
            pe_ped_top_grad.data,
            pe_background.data
        ])
        
        # EPM: EFIT++

        #neutral beam, Ip, toroidal magnetic field, stored energy, beta_N

        Ip        = self.Ip
        #toroidal magnetic field (at axis)
        Btor      = self.Btor

        # storedEnergy (joules)
        plasmaEnergy = self.plasmaEnergy

        rmaxis    = self.rmaxis
        zmaxis    = self.zmaxis
        rbdy      = self.rbdy
        zbdy      = self.zbdy
        rmidin    = self.rmidin
        rmidout   = self.rmidout
        aminor    = self.aminor
        kappa     = self.kappa
        deltaup   = self.deltaup
        deltalow  = self.deltalow
        #pmaxis    = self.pmaxisx
        #psibdy    = self.psibdy

        rprof     = self.rprof
        psinprof  = self.psinprof

        r_2D      = self.r_2D
        z_2D      = self.z_2D
        psin_2D   = self.psin_2D
        #psi_2D    = self.psi_2D
        #in megawatts

        ultimatemintime = 0.1
        mintime   = numpy.max([numpy.min(times_apf),numpy.min(times_epm),ultimatemintime])
        maxtime   = numpy.min([numpy.max(times_apf),numpy.max(times_epm)])
        time_index = numpy.where((times_apf >= mintime) & (times_apf <= maxtime))[0]
        times0    = numpy.array(times_apf[time_index])

        te       = self.te 
        dte      = self.dte 
        ne       = self.ne  
        dne      = self.dne 
        r        = self.r  
        psinprof = self.psinprof 
        rprof    = self.rprof    
        Ip       = self.Ip 

        shot = self.shotNum
                
        # First check if the rprof data from epm is good (at least two points > rmaxis exist). This filters out if rprof is all nans
        
        # Set up times array
        times = []
        
        if presetTimes != []:
            times = presetTimes
        else:
            for j in range(0,len(times0)):

                test_index_apf = numpy.argmin(abs(times_apf-times0[j]))
                test_index_epm = numpy.argmin(abs(times_epm-times_apf[test_index_apf]))
                index          = numpy.where(rprof.data[test_index_epm]>rmaxis.data[test_index_epm])[0]
                if len(index)  > 2:
                    times.append(times0[j])


        # Some default time slices
        #times = [0.173,0.456]
    #    times = [0.3828,0.3889,0.3949]

        Aratio     = numpy.zeros(len(times))
        elong      = numpy.zeros(len(times))
        delta      = numpy.zeros(len(times))

        W_ped      = numpy.zeros(len(times))
        beta_ped   = numpy.zeros(len(times))

        W_ped_psin_te  = numpy.zeros(len(times))
        H_ped_psin_te  = numpy.zeros(len(times))
        W_ped_psin_ne  = numpy.zeros(len(times))
        H_ped_psin_ne  = numpy.zeros(len(times))
        W_ped_psin_pe  = numpy.zeros(len(times))
        H_ped_psin_pe  = numpy.zeros(len(times))

        W_ped_radius_te  = numpy.zeros(len(times))
        H_ped_radius_te  = numpy.zeros(len(times))
        W_ped_radius_ne  = numpy.zeros(len(times))
        H_ped_radius_ne  = numpy.zeros(len(times))
        W_ped_radius_pe  = numpy.zeros(len(times))
        H_ped_radius_pe  = numpy.zeros(len(times))


        IpAdj=[]
        IpMax = np.ones(len(times))
        shotIndexed = []
        NBIAdj=[]
        ssNBIAdj = []
        swNBIAdj = []
        betaNAdj = []

        # For each time slice
        for i in range(0,len(times)):
            time = times[i]

            IpAdj += [self.Ip[np.argmin(np.abs(self.IpTime-time))]]
            shotIndexed += [int(self.shotNum)]
            if self.total_NBI_power != None:
                NBIAdj += [self.total_NBI_power.data[np.argmin(np.abs(self.total_NBI_power.time.data-time))]]
            else:
                NBIAdj += [0]
            if self.ss_NBI_power != None:
                ssNBIAdj += [self.ss_NBI_power.data[np.argmin(np.abs(self.ss_NBI_power.time.data-time))]]
            else:
                ssNBIAdj += [0]
            if self.sw_NBI_power != None:
                swNBIAdj += [self.sw_NBI_power.data[np.argmin(np.abs(self.sw_NBI_power.time.data-time))]]
            else:
                swNBIAdj += [0]

            betaNAdj += [self.betaN.data[np.argmin(np.abs(self.betaN.time.data-time))]]
            
            time_index_apf = numpy.argmin(abs(times_apf-time))
            if plotVsRadius or plotVsPsiN:
                time_index_ayc = numpy.argmin(abs(times_ayc-times_apf[time_index_apf]))
            time_index_epm = numpy.argmin(abs(times_epm-times_apf[time_index_apf]))

            if printTimes:
                print("Time = ",times_apf[time_index_apf])
                #if plotVsRadius:
                #    print("Time = ",times_ayc[time_index_ayc])
                #print("Time = ",times_epm[time_index_epm])

            index       = numpy.where(rprof.data[time_index_epm]>rmaxis.data[time_index_epm])[0] 
            
            radius_efit = numpy.array(rprof.data[time_index_epm][index])
            psin_efit   = numpy.array(psinprof.data[time_index_epm][index])

            # specify a radius axis on which to evaluate mtanh
            redge  = radius_efit[-1]
            r0     = redge - 0.1 #1.3
            r1     = redge + 0.1 #1.45
            npnts  = 256
            radius = np.linspace(r0, r1, npnts)

            index2 = numpy.where(radius<=redge)[0]
            psin   = numpy.interp(radius[index2],radius_efit,psin_efit)
            interp = scipy.interpolate.RegularGridInterpolator((r_2D.data,z_2D.data),psin_2D.data[time_index_epm])
            psin2  = interp(numpy.array([[i,zmaxis.data[time_index_epm]] for i in radius]))


            # create an instance of the model class
            model = mtanh()
            # get the mtanh predictions of the temperature and density
            te_profile = model.prediction(radius, te_parameters[:, time_index_apf])
            ne_profile = model.prediction(radius, ne_parameters[:, time_index_apf])
            pe_profile = model.prediction(radius, pe_parameters[:, time_index_apf])

            rped_te = te_ped_location.data[time_index_apf]
            wped_te = te_ped_width.data[time_index_apf]
            teped   = te_ped_height.data[time_index_apf]
            rped_te_top = rped_te-0.5*wped_te
            rped_te_bot = rped_te+0.5*wped_te
            psin_ped_te = numpy.interp(rped_te,radius[index2],psin)
            psin_ped_te_top = numpy.interp(rped_te_top,radius[index2],psin)
            if rped_te_bot < redge:
                psin_ped_te_bot = numpy.interp(rped_te_bot,radius[index2],psin)
            else:
                psin_ped_te_bot = numpy.interp(rped_te_bot,radius,psin2)

            rped_ne = ne_ped_location.data[time_index_apf]
            wped_ne = ne_ped_width.data[time_index_apf]
            neped   = ne_ped_height.data[time_index_apf]/1e19
            rped_ne_top = rped_ne-0.5*wped_ne
            rped_ne_bot = rped_ne+0.5*wped_ne
            psin_ped_ne = numpy.interp(rped_ne,radius[index2],psin)
            psin_ped_ne_top = numpy.interp(rped_ne_top,radius[index2],psin)
            if rped_ne_bot < redge:
                psin_ped_ne_bot = numpy.interp(rped_ne_bot,radius[index2],psin)
            else:
                psin_ped_ne_bot = numpy.interp(rped_ne_bot,radius,psin2)

            rped_pe = pe_ped_location.data[time_index_apf]
            wped_pe = pe_ped_width.data[time_index_apf]
            peped   = pe_ped_height.data[time_index_apf]/1000.0
            rped_pe_top = rped_pe-0.5*wped_pe
            rped_pe_bot = rped_pe+0.5*wped_pe
            psin_ped_pe = numpy.interp(rped_pe,radius[index2],psin)
            psin_ped_pe_top = numpy.interp(rped_pe_top,radius[index2],psin)
            if rped_pe_bot < redge:
                psin_ped_pe_bot = numpy.interp(rped_pe_bot,radius[index2],psin)
            else:
                psin_ped_pe_bot = numpy.interp(rped_pe_bot,radius,psin2)

            # Calculate and print quantities

            psin_mid     = (psin_ped_te+psin_ped_ne)/2.0
            wped_psin_te = psin_ped_te_bot - psin_ped_te_top
            wped_psin_ne = psin_ped_ne_bot - psin_ped_ne_top
            wped_psin_pe = psin_ped_pe_bot - psin_ped_pe_top
            wped_psin    = (wped_psin_te+wped_psin_ne)/2.0
            psin_ped_top = psin_mid - wped_psin/2.0
            psin_ped_bot = psin_mid + wped_psin/2.0
            pped         = 2.0*numpy.interp(psin_ped_top,psin,pe_profile[index2]/1000.0)
            mu0          = 4.0*numpy.pi*1.0e-7
            r_lcfs       = numpy.array(rbdy.data[time_index_epm])
            z_lcfs       = numpy.array(zbdy.data[time_index_epm])
            dist_array   = (r_lcfs[:-1]-r_lcfs[1:])**2.0 + (z_lcfs[:-1]-z_lcfs[1:])**2.0
            l_lcfs       = numpy.sum(numpy.sqrt(dist_array))
            Bpol         = mu0*Ip.data[time_index_epm]/l_lcfs
            beta         = 2.0*mu0*pped*1000.0/(Bpol**2.0)

            rmi          = numpy.array(rmidin.data[time_index_epm]) 
            rmo          = numpy.array(rmidout.data[time_index_epm])    
            ami          = numpy.array(aminor.data[time_index_epm])   
            kap          = numpy.array(kappa.data[time_index_epm])      
            dlu          = numpy.array(deltaup.data[time_index_epm])    
            dll          = numpy.array(deltalow.data[time_index_epm])   
            
            Aratio[i]    = (rmi+rmo)/(2.0*ami)
            elong[i]     = kap
            delta[i]     = (dlu+dll)/2.0


            W_ped[i]     = wped_psin
            beta_ped[i]       = beta

            W_ped_psin_te[i]  = wped_psin_te
            H_ped_psin_te[i]  = numpy.interp(psin_ped_te_top,psin,te_profile[index2])
            W_ped_psin_ne[i]  = wped_psin_ne
            H_ped_psin_ne[i]  = numpy.interp(psin_ped_ne_top,psin,ne_profile[index2])
            W_ped_psin_pe[i]  = wped_psin_pe
            H_ped_psin_pe[i]  = numpy.interp(psin_ped_pe_top,psin,pe_profile[index2])

            W_ped_radius_te[i]  = wped_te
            H_ped_radius_te[i]  = numpy.interp(rped_te_top,radius,te_profile)
            W_ped_radius_ne[i]  = wped_ne
            H_ped_radius_ne[i]  = numpy.interp(rped_ne_top,radius,ne_profile)
            W_ped_radius_pe[i]  = wped_pe
            H_ped_radius_pe[i]  = numpy.interp(rped_pe_top,radius,pe_profile)


            if printQuantities:

                print("")
                #print("Psin_mid = ",psin_mid)
                #print("Psin_ped = ",psin_ped_top)
                #print("pped     = ",pped)
                #print("Time     = ",times_apf[time_index_apf])
                print("W_ped    = ",wped_psin)
                print("beta_ped = ",beta)
                print("W", W_ped_psin_te)
                print("H_ped_psin_te/1000.0", H_ped_psin_te/1000.0)
                print("W_ped_psin_ne", W_ped_psin_ne)
                print("H_ped_psin_ne/1.0e20", H_ped_psin_ne/1.0e20)

                print("")

            fs = 16

            if plotVsRadius:
            
                # Plot the profiles at a given time, vs. radius


                fig = plt.figure(figsize=(6,10))
                ax1 = fig.add_subplot(3, 1, 1)
                ax2 = fig.add_subplot(3, 1, 2)
                ax3 = fig.add_subplot(3, 1, 3)
                fig.suptitle(f"{shot} @ {te_ped_location.time.data[time_index_apf]:.3f} ms",fontsize=fs)

                ax1.plot(radius, te_profile, lw=2, color="red")
                ax1.errorbar(r.data[time_index_ayc],te.data[time_index_ayc],yerr=dte.data[time_index_ayc],color='blue',marker='o',linestyle='None')
                ax1.plot((rped_te,rped_te),(0.0,teped), lw=2, color='black', linestyle='--')
                ax1.plot((rped_te_top,rped_te_top),(0.0,teped), lw=2, color='black', linestyle=':')
                ax1.plot((rped_te_bot,rped_te_bot),(0.0,teped), lw=2, color='black', linestyle=':')
                ax1.set_xlabel("")
                ax1.set_ylabel("$T_{e}$ (eV)",fontsize=fs)
                ax1.set_ylim([0, 400])
                # ax1.set_ylim([0.,1.20*numpy.max(te_profile)])
                ax1.set_xlim([r0-0.10*(r1-r0),r1+0.10*(r1-r0)])
                ax1.tick_params(axis='x',labelsize=fs)
                ax1.tick_params(axis='y',labelsize=fs)
                ax1.tick_params(labelbottom=False)

                ax2.plot(radius, ne_profile/1e19, lw=2, color="red")
                ax2.errorbar(r.data[time_index_ayc],ne.data[time_index_ayc]/1e19,yerr=dne.data[time_index_ayc]/1e19,color='blue',marker='o',linestyle='None')
                ax2.plot((rped_ne,rped_ne),(0.0,neped), lw=2, color='black', linestyle='--')
                ax2.plot((rped_ne_top,rped_ne_top),(0.0,neped), lw=2, color='black', linestyle=':')
                ax2.plot((rped_ne_bot,rped_ne_bot),(0.0,neped), lw=2, color='black', linestyle=':')
                ax2.set_ylabel("$n_{e}$ ($10^{19}$ m$^{-3}$)",fontsize=fs)
                # ax2.set_ylim([0,1.20*numpy.max(ne_profile/1e19)])
                ax2.set_ylim([0, 6])
                ax2.set_xlim([r0-0.10*(r1-r0),r1+0.10*(r1-r0)])
                ax2.tick_params(axis='x',labelsize=fs)
                ax2.tick_params(axis='y',labelsize=fs)
                ax2.tick_params(labelbottom=False)

                ax3.plot(radius, pe_profile/1000.0, lw=2, color="red")
                ax3.plot((rped_pe,rped_pe),(0.0,peped), lw=2, color='black', linestyle='--')
                ax3.plot((rped_pe_top,rped_pe_top),(0.0,peped), lw=2, color='black', linestyle=':')
                ax3.plot((rped_pe_bot,rped_pe_bot),(0.0,peped), lw=2, color='black', linestyle=':')
                ax3.set_xlabel("$R$ (m)",fontsize=fs)
                ax3.set_ylabel("$p_{e}$ (kPa)",fontsize=fs)
                ax3.set_ylim([0.,1.20*numpy.max(pe_profile/1000.0)])
                ax3.set_xlim([r0-0.10*(r1-r0),r1+0.10*(r1-r0)])
                ax3.tick_params(axis='x',labelsize=fs)
                ax3.tick_params(axis='y',labelsize=fs)

                plt.tight_layout()
                if saveFigure:
                    plt.savefig("plots/"+self.shotNum+"_"+str(int(time*1000))+"_"+"plotVsRadius.png")
                    print("saved " + "plots/"+self.shotNum+"_"+str(int(time*1000))+"_"+"plotVsRadius.png")
                if showFigure:
                    plt.show()


            if plotVsPsiN:
            
                # Plot the profiles at a given time, vs. psin

                fig = plt.figure(figsize=(6,10))
                ax1 = fig.add_subplot(3, 1, 1)
                ax2 = fig.add_subplot(3, 1, 2)
                ax3 = fig.add_subplot(3, 1, 3)
                fig.suptitle(f"{shot} @ {te_ped_location.time.data[time_index_apf]:.3f} ms",fontsize=fs)

                ax1.plot(psin, te_profile[index2], lw=2, color="red")
                ymax = 1.20*numpy.max(te_profile)
                ax1.plot((psin_ped_te,psin_ped_te),(0.0,teped), lw=2, color='black', linestyle='--')
                ax1.plot((psin_ped_te_top,psin_ped_te_top),(0.0,teped), lw=2, color='black', linestyle=':')
                ax1.plot((psin_ped_te_bot,psin_ped_te_bot),(0.0,teped), lw=2, color='black', linestyle=':')
                ax1.errorbar(np.interp(r.data[time_index_ayc],radius[index2],psin),te.data[time_index_ayc],yerr=dte.data[time_index_ayc],color='blue',marker='o',linestyle='None')

                # ax1.plot((psin_mid,psin_mid),(0.0,ymax), lw=2, color='blue', linestyle='--')
                # ax1.plot((psin_ped_top,psin_ped_top),(0.0,ymax), lw=2, color='blue', linestyle=':')
                # ax1.plot((psin_ped_bot,psin_ped_bot),(0.0,ymax), lw=2, color='blue', linestyle=':')
                # ax1.set_xlabel("")
                ax1.set_ylabel("$T_{e}$ (eV)",fontsize=fs)
                ax1.set_ylim([0.0, ymax])
                # ax1.set_ylim(psiN_ylim)
                ax1.set_xlim([0.75, 1])
                # ax1.set_xlim(psiN_xlim)
                ax1.tick_params(axis='x',labelsize=fs)
                ax1.tick_params(axis='y',labelsize=fs)
                ax1.tick_params(labelbottom=False)


                ax2.plot(psin, ne_profile[index2]/1e19, lw=2, color="red")
                ymax = 1.20*numpy.max(ne_profile/1e19)
                ax2.errorbar(np.interp(r.data[time_index_ayc],radius[index2],psin),ne.data[time_index_ayc]/1e19,yerr=dne.data[time_index_ayc]/1e19,color='blue',marker='o',linestyle='None')

                ax2.plot((psin_ped_ne,psin_ped_ne),(0.0,neped), lw=2, color='black', linestyle='--')
                ax2.plot((psin_ped_ne_top,psin_ped_ne_top),(0.0,neped), lw=2, color='black', linestyle=':')
                ax2.plot((psin_ped_ne_bot,psin_ped_ne_bot),(0.0,neped), lw=2, color='black', linestyle=':')
                # ax2.plot((psin_mid,psin_mid),(0.0,ymax), lw=2, color='blue', linestyle='--')
                # ax2.plot((psin_ped_top,psin_ped_top),(0.0,ymax), lw=2, color='blue', linestyle=':')
                # ax2.plot((psin_ped_bot,psin_ped_bot),(0.0,ymax), lw=2, color='blue', linestyle=':')
                ax2.set_ylabel("$n_{e}$ ($10^{19}$ m$^{-3}$)",fontsize=fs)
                #ax2.set_ylim([0.0, ymax])
                ax2.set_ylim([0.0, 6.0])
                #ax2.set_xlim([0.9, 1.05])
                ax2.set_xlim([0.75, 1])
                ax2.tick_params(axis='x',labelsize=fs)
                ax2.tick_params(axis='y',labelsize=fs)
                ax2.tick_params(labelbottom=False)

                ax3.plot(psin, pe_profile[index2]/1000.0, lw=2, color="red")
                ymax = 1.20*numpy.max(pe_profile/1000.0)
                #ax3.plot((psin_ped_pe,psin_ped_pe),(0.0,peped), lw=2, color='black', linestyle='--')
                #ax3.plot((psin_ped_pe_top,psin_ped_pe_top),(0.0,peped), lw=2, color='black', linestyle=':')
                #ax3.plot((psin_ped_pe_bot,psin_ped_pe_bot),(0.0,peped), lw=2, color='black', linestyle=':')
                ax3.plot((psin_mid,psin_mid),(0.0,ymax), lw=2, color='blue', linestyle='--')
                ax3.plot((psin_ped_top,psin_ped_top),(0.0,ymax), lw=2, color='blue', linestyle=':')
                ax3.plot((psin_ped_bot,psin_ped_bot),(0.0,ymax), lw=2, color='blue', linestyle=':')
                ax3.set_xlabel("$\Psi_{N}$",fontsize=fs)
                ax3.set_ylabel("$p_{e}$ (kPa)",fontsize=fs)
                #ax3.set_ylim([0.0, ymax])
                ax3.set_ylim([0.0, 2.0])
                #ax3.set_xlim([0.9, 1.05])
                ax3.set_xlim([0.75, 1])
                ax3.tick_params(axis='x',labelsize=fs)
                ax3.tick_params(axis='y',labelsize=fs)

                plt.tight_layout()
                if saveFigure:
                    plt.savefig("plots/"+self.shotNum+"_"+str(int(time*1000))+"_"+"plotVsPsiN.png")
                    print("saved " + "plots/"+self.shotNum+"_"+str(int(time*1000))+"_"+"plotVsPsiN.png")
                if showFigure:
                    plt.show()

        IpMax *= np.max(IpAdj)
                
        if plotVsTime:

            # Plot pedestal radial locations vs. time

    #        plt.plot(te_ped_location.time.data, te_ped_location.data, ".-", label="temperature location")
    #        plt.plot(ne_ped_location.time.data, ne_ped_location.data, ".-", label="density location")
    #        plt.xlabel("Time (s)")
    #        plt.ylabel("Pedestal radial position (m)")
    #        plt.legend()
    #        plt.grid()
    #        plt.tight_layout()
    #        plt.show()
        
            plt.plot(times, W_ped, label="Width")
            plt.plot(times, beta_ped, label="Beta")
            plt.plot(times, W_ped_psin_te, label="Width_Te_psin")
            #plt.plot(times, H_ped_psin_te, label="Teped_psin")
            plt.xlabel("Time (s)")
            plt.ylabel("Pedestal quantities")
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.plot(W_ped, beta_ped, label="Beta")
            plt.plot(W_ped_psin_te, H_ped_psin_te, label="Beta")
            plt.xlabel("W_ped")
            plt.ylabel("Beta_ped")
            plt.tight_layout()
            plt.show()

        
        if savePklForShot:

            pkldata = {'shot': shot, 'times': times, 'W_ped': W_ped, 'Beta_ped': beta_ped,
                    'W_ped_psin_te': W_ped_psin_te,'W_ped_psin_ne': W_ped_psin_ne,'W_ped_psin_pe': W_ped_psin_pe,
                    'H_ped_psin_te': H_ped_psin_te,'H_ped_psin_ne': H_ped_psin_ne,'H_ped_psin_pe': H_ped_psin_pe,
                    'W_ped_radius_te': W_ped_radius_te,'W_ped_radius_ne': W_ped_radius_ne,'W_ped_radius_pe': W_ped_radius_pe,
                    'H_ped_radius_te': H_ped_radius_te,'H_ped_radius_ne': H_ped_radius_ne,'H_ped_radius_pe': H_ped_radius_pe,
                    'aratio': Aratio, 'elong': elong, 'delta': delta, 'NBI': NBIAdj, 'ssNBI':ssNBIAdj, "swNBI":swNBIAdj,
                    "IpMax":IpMax, "Ip":IpAdj, "shotIndexed":shotIndexed, "betaN":betaNAdj}
            filename = 'outputWithBeamPower3/MAST-U_pedestal_'+str(shot)+'.pkl'
            outfile = open(filename, 'wb')
            pickle.dump(pkldata,outfile)
            outfile.close()
            print('outputWithBeamPower3/MAST-U_pedestal_'+str(shot)+'.pkl saved')



    def contourPlot(self, plotnumber, saveFigure=False, showFigure=True, fitHMode=False, plotName = "default", numPix = 30,
                    cbarMax ="default", cbarMin="default", numMin = 20, countType = "count", IpMin = 0.9, lowSlopeFilter = 0.8, beamNumber = "twoBeams", elongRange="all", elmRange="all"):
        """Generates a contour plot comparing two parameters, and coloring by a third parameter.

        Args:
            plotnumber (int): Determines the x/y axes.\n
            1 for Beta vs. Delta\n
            2 for Te,ped vs. Delta_te\n
            3 for ne,ped vs. Delta_ne\n
            4 for pe,ped vs. Delta_pe\n
            5 for  te,ped,r vs. W_r_te\n
            6 for ne,ped,r vs. W_r_ne\n
            7 for pe,ped,r vs. W_r_pe]\n
            8 for delta vs kappa\n
            9 for aratio vs delta\n
            10 for elong vs delta\n
            11 for ne vs te
            saveFigure (bool, optional): Saves the plot to "plots/{shot # or 'allShots'}{typeOfContourPlot}Colored{countType}.png.\n
            Name overriden by plotName kwarg. Defaults to False.
            showFigure (bool, optional): Displays figure in matplotlib window. Defaults to True.
            fitHMode (bool, optional): _description_. Defaults to False.
            plotName (str, optional): Overrides default title for saveFigure. Defaults to {see saveFigure}.
            numPix (int, optional): Number of pixels in the x and y directions. Defaults to 60.
            cbarMax (int, optional): Maximum value of the colorbar. Values above this value will take on the max valued color. Defaults to 2.
            cbarMin (int, optional): Minimum value of the colorbar. Values below this value will be white. Defaults to 0.
            numMin (int, optional): Minimum number of equilibria in a pixel for the pixel to show up. Defaults to 10.
            posPed(bool, optional): Removes points with negative pedestal height. Defaults to True.
            countType (str, optional): Determines what parameter will be colored by the colorbar.\n
            "count" - log10 number of equilibria in the pixel\n
            "elong" - average elongation of the equilibria in the pixel\n
            "delta" - "" but with triangularity\n
            "beta" - "" but with pedestal height\n
            "slope" - "" but with pedestal slope\n
            Defaults to "count".\n
            IpMin (float, optional): Equilibria with plasma current below this percentage of the maximum will be filtered out. Defaults to 0.9.
            lowSlopeFilter (bool, optional): Filters equilibria with beta/delta < 0.75. Defaults to True.
        
         Adapted from Jack Berkery contourPlot 2024
         """
        
        # Data for contour is pulled from a pkl
        if not self.pkl:
            raise Exception("Must have pkl data to run contourPlot")
        
        cbarMinDict = {"count":0 if numMin<1 else np.log10(numMin),
                        "elong":1.95,
                        "delta":0.45,
                        "pedestalHeight":0,
                        "pedestalSlope":0.75}
        cbarMaxDict = {"count":2,
                        "elong":2.15,
                        "delta":0.55,
                        "pedestalHeight":0.2,
                        "pedestalSlope":3.75}
        if cbarMin == "default":
            cbarMin = cbarMinDict[countType]
        if cbarMax == "default":
            cbarMax = cbarMaxDict[countType]
        def setupfigure(figurenumber,xsize,ysize):
            '''Setup plotspace'''
            figurename = plt.figure(figurenumber,figsize=(xsize,ysize),
                                    edgecolor='white')
            matplotlib.rcParams['xtick.major.pad'] = 8
            matplotlib.rcParams['ytick.major.pad'] = 8
            return figurename
        def setupframe(framecolumns,framerows,position,x1,x2,y1,y2,numberofxticks,
                    numberofyticks,xlabel,ylabel,xminor,yminor,font_size):
            '''Setup plot'''
            framename = plt.subplot(framerows,framecolumns,position)
            plt.axis([x1,x2,y1,y2])
            xtickarray = []
            for i in range(0,numberofxticks+1):
                xtickarray.append(float(i)/numberofxticks*(x2-x1)+x1)
            plt.xticks(xtickarray,fontsize=font_size)
            ytickarray = []
            for i in range(0,numberofyticks+1):
                ytickarray.append(float(i)/numberofyticks*(y2-y1)+y1)
            plt.yticks(ytickarray,fontsize=font_size)
            plt.ylabel(ylabel,fontsize=font_size)
            plt.tight_layout()
            if xlabel=='':
                framename.axes.xaxis.set_ticklabels([])
            else:
                plt.xlabel(xlabel,fontsize=font_size)
            if xminor!=0.0:
                framename.axes.xaxis.set_minor_locator(MultipleLocator(xminor))
            if yminor!=0.0:
                framename.axes.yaxis.set_minor_locator(MultipleLocator(yminor))
            return framename
        def makefigure(*args):
            '''Generates contour plot'''
            figure1 = setupfigure(1,6.0,5.0)
            font_size = 20

            xx = np.linspace(x1,x2,num=xsize+1)
            yy = np.linspace(y1,y2,num=ysize+1)

            xx2 = np.delete(xx+(xx[1]-xx[0])/2.0,-1)
            yy2 = np.delete(yy+(yy[1]-yy[0])/2.0,-1)

            Ntot   = np.zeros((len(xx)-1,len(yy)-1))
            totalPoints = 0
            NBIMin = {"noBeams" : 0,
                      "oneBeam" : 0.5,
                      "twoBeams" : 2.25,
                      "all" : 0}
            NBIMax = {"noBeams" : 0.5,
                      "oneBeam" : 2.25,
                      "twoBeams" : np.inf, 
                      "all" : np.inf}
            elongMin = {"low" : 0,
                    "med" : 1.99,
                    "hi" : 2.105,
                    "all" : 0}
            elongMax = {"low" :1.99,
                    "med" : 2.105,
                    "hi" : np.inf, 
                    "all" : np.inf}
            elmMin = {"early" : 0,
                      "mid" : 0.33,
                      "late" : 0.66,
                      "all" : 0}
            elmMax = {"early" :0.33,
                      "mid" : 0.66,
                      "late" : 1, 
                      "all" : 1}
            # Counts number of points that lie within each bin
            totalIndex =[]
            for i in range(0,len(xx)-1):
                for j in range(0,len(yy)-1):
                    #Can add additional conditions here to filter what is shown in contour
                    index, = np.where((xquantity>=xx[i])   & 
                                        (xquantity< xx[i+1]) &
                                        (yquantity>=yy[j])   &
                                        (yquantity< yy[j+1]) &
                                        (self.Ip>IpMin*self.IpMax) &
                                        (self.Beta_ped/self.W_ped >lowSlopeFilter)&
                                        (self.NBI > NBIMin[beamNumber]) &
                                        (self.NBI < NBIMax[beamNumber]) & 
                                        (self.elong > elongMin[elongRange]) &
                                        (self.elong < elongMax[elongRange]) & 
                                        (self.elmPercent > elmMin[elmRange]) &
                                        (self.elmPercent < elmMax[elmRange]) )
                    totalIndex += list(index)
                    if len(index) >= numMin:
                        totalPoints += len(index)

                        if countType == "count":
                            if len(index) == 0:
                                Ntot[i,j] = -1e-10
                            else:
                                Ntot[i,j] = np.log10(len(index))
                        elif countType == 'elong':
                            Ntot[i,j]   = np.median(self.elong[index])
                        elif countType == "delta":
                            Ntot[i,j] = np.median(self.delta[index])
                        elif countType == "time":
                            Ntot[i,j] = np.median(self.times[index])
                        elif countType == "aratio":
                            Ntot[i,j] = np.median(self.aratio[index])
                        elif countType == "pedestalHeight":
                            Ntot[i,j] = np.median(self.Beta_ped[index])
                        elif countType == "pedestalSlope":
                            Ntot[i,j] = np.median(self.Beta_ped[index]/self.W_ped[index])


                    else:
                        Ntot[i,j] = None
            print(totalIndex)
            print(len(totalIndex))
            if (countType == "time"):
                zeroindex = np.where(Ntot == 0.0)
                Ntot[zeroindex] = -1.0e-10
            zz = np.transpose(Ntot)
            self.zz = zz

            frame1 = setupframe(1,1,1,x1,x2,y1,y2,
                                xticks,yticks,xlabel,
                                ylabel,xminor,yminor,font_size)
            
            CS = plt.imshow(zz,extent=(x1,x2,y1,y2),origin='lower',
                            interpolation='none',
                            aspect='auto',
                            vmin=cbarMin,vmax=cbarMax)    
            cbar = plt.colorbar(CS,ticks=[cbarMin, (cbarMax-cbarMin)/2+cbarMin, cbarMax])#,3.0])
            colorDict = {
                "count":"viridis",
                "elong":"rainbow",
                "delta":"plasma",
                "pedestalHeight":"rainbow",
                "pedestalSlope":"plasma"
            }

            try:
                cmap = copy.copy(matplotlib.colormaps.get_cmap(colorDict[countType]))
            except:
                cmap = copy.copy(matplotlib.cm.get_cmap(colorDict[countType]))

            cmap.set_under(color='white')
            cmap.set_bad(color='white')
            plt.set_cmap(cmap)
            cbar.ax.set_yticklabels([str(cbarMin), str(np.round((cbarMax-cbarMin)/2+cbarMin, 3)),str(cbarMax)],fontsize=font_size)#,'3.0'],fontsize=font_size)
            if countType=="count":
                cbar.ax.set_ylabel('Log$_{10}$ (Number of equilibria)',fontsize=font_size)
            elif countType =="elong":
                cbar.ax.set_ylabel(r'Median $\kappa$',fontsize=font_size)
            elif countType == "delta":
                cbar.ax.set_ylabel(r'Median $\delta$',fontsize=font_size)
            elif countType == "time":
                cbar.ax.set_ylabel(r'Median Time',fontsize=font_size)
            elif countType == "aratio":
                cbar.ax.set_ylabel(r'Aspect Ratio',fontsize=font_size)




            plt.subplots_adjust(left=0.20,right = 0.90,bottom=0.20,top=0.92)
            # This provides a square plot area with a 5in by 6in figure area and the colorbar on the right
            if plotnumber == 1 or plotnumber == 0:
                x_width = np.linspace(x1,x2,100)
                y_beta  = (x_width/0.43)**(1.0/1.03)
                y_beta2 = (x_width/0.08)**(2.0)

                #Plots some predictive models on top of contour plot
                #plt.plot(x_width,y_beta,color='red',linestyle='--')
                #plt.plot(x_width,y_beta2,color='magenta',linestyle='--')
                #plt.plot(x_width, 3/4*x_width, color="blue")
                #plt.annotate(r'NSTX GCP: $\Delta_{\mathrm{ped}} = 0.43\beta_{\theta,\mathrm{ped}}^{1.03}$',(0.06,0.308),color='red',fontsize=13,annotation_clip=False)
                #plt.annotate(r'$\Delta_{\mathrm{ped}} = 0.43\beta_{\theta,\mathrm{ped}}^{1.03}$',(0.12,y2+0.008),color='red',fontsize=13,annotation_clip=False)
                #plt.annotate(r'$\Delta_{\mathrm{ped}} = 0.08\beta_{\theta,\mathrm{ped}}^{0.5}$',(0.0,y2+0.008),color='magenta',fontsize=13,annotation_clip=False)
                # plt.plot(x_width, 0.75*x_width, "r")
                #Provides a fit to the HMode data
                
                if fitHMode:
                    validXQuantity = np.array([])
                    validYQuantity = np.array([])
                    # from smith paper
                    # plt.plot(x_width, 0.146*np.sqrt(x_width), label=r"$0.146\sqrt{\beta}$")
                    # plt.plot(x_width, 0.104*x_width**0.309, label=r"$0.104\beta^{0.309}$")

                    for i in range(len(xquantity)):
                        #filters which data points to fit
                        if((x1<xquantity[i]<x2) and 
                        (y1<yquantity[i]<y2) and 
                        # right now, Lmode is considered to be below a horizontal line with slope 0.75 (in BetavsDelta)
                        (yquantity[i]>0.75*xquantity[i])):
                            validXQuantity = np.append(validXQuantity,xquantity[i])
                            validYQuantity = np.append(validYQuantity, yquantity[i])
                #    plt.close()
                    # Define the function to fit.
                    # The first parameter is the independent variable. # The remaining parameters␣are the fit parameters.
                    
                    # Function to fit
                    def curve(x,m):
                        return  m*x
                    # Do the fit. Read the help on curve_fit!
                    (popt,pcov) = curve_fit(curve, validXQuantity, validYQuantity, p0=(20))
                    # popt now holds the optimized parameters a in popt[0] and b in popt[1]
                    # pcov is the covariance matrix, which gives errors and correlations.
                    # To extract just the errors on the fit parameters as sigmas:
                    popt = np.round(popt, 3)
                    perr = np.round(np.sqrt(np.diag(pcov)), 3)
                    # now perr holds the +- 1sigma error for a and b.
                    # First plot the data with error bars
                    #plt.plot(xquantity, yquantity, "o", label="data", color="black")
                    # There are many ways to plot the line given a and b, but here’s one:
                    yfit = curve(x_width, *popt) # * passes a list as remaining function parameters

                    # plots the fit curve
                    plt.plot(x_width, yfit,label=f"fit - {popt[0]}$\pm${perr[0]} $\delta$", color="black")

                    #Plots the points that were fitted
                    plt.plot(validXQuantity, validYQuantity, ".", markersize = 0.5, color="red")
                    plt.legend()
                    # savefig("hw1_meaningful_file_name.pdf")
                    # PDFs can be used as LaTeX figures
            # elif plotnumber == 3:
                # plt.vlines(0.015, 0, 0.6)
                # plt.vlines(0.01, 0, 0.6)
            if saveFigure:
                if plotName == "default":
                    plt.savefig("plots/"+outfilename+'.png')
                    print("saved "+"plots/"+outfilename+'.png')
                else:
                    plt.savefig("plots/"+plotName+'.png')
                    print("saved "+"plots/"+plotName+'.png')

            if showFigure:
                plt.title(str(totalPoints)+" equilibria")
                plt.legend()
                plt.show()        # Beta vs. Delta
        
        if plotnumber == 0:

            outfilename = "deltaVsBeta"

            yquantity    = self.W_ped
            ylabel       = r'$\Delta_{\mathrm{ped}}$'
            y1           = 0.0
            y2           = 0.15
            yticks       = 4
            yminor       = 0.025
            ysize        = numPix

            xquantity    = self.Beta_ped
            xlabel       = r'$\beta_{\theta,\mathrm{ped}}$'
            x1           = 0.0
            x2           = 0.5
            xticks       = 4
            xminor       = 0.025
            xsize        = numPix

        # Te,ped vs. Delta_te
        
        if plotnumber == 1:

            outfilename = "betavsdelta"

            xquantity    = self.W_ped
            xlabel       = r'$\Delta_{\mathrm{ped}}$'
            x1           = 0
            x2           = 0.12
            xticks       = 3
            xminor       = 0.025
            xsize        = numPix

            yquantity    = self.Beta_ped
            ylabel       = r'$\beta_{\theta,\mathrm{ped}}$'
            y1           = 0
            y2           =0.3
            yticks       = 3
            yminor       = 0.025
            ysize        = numPix

        # Te,ped vs. Delta_te
    
        if plotnumber == 2:

            outfilename = "tevsdelta"

            xquantity    = self.W_ped_psin_te
            xlabel       = r'$\Delta_{\mathrm{ped,Te}}$'
            x1           = 0.0
            x2           = 0.15
            xticks       = 3
            xminor       = 0.025
            xsize        = numPix

            yquantity    = self.H_ped_psin_te/1000.0
            ylabel       = r'$T_{\mathrm{e,ped}}$ (keV)'
            y1           = 0.0
            y2           = 0.4
            yticks       = 4
            yminor       = 0.05
            ysize        = numPix

        # ne,ped vs. Delta_ne

        if plotnumber == 3:

            outfilename = "nevsdelta"

            xquantity    = self.W_ped_psin_ne
            xlabel       = r'$\Delta_{\mathrm{ped,ne}}$'
            x1           = 0.0
            x2           = 0.15
            xticks       = 3
            xminor       = 0.05
            xsize        = numPix

            yquantity    = self.H_ped_psin_ne/1.0e20
            ylabel       = r'$n_{\mathrm{e,ped}}$ ($10^{20}$ m$^{-3}$)'
            y1           = 0
            y2           = 0.6
            yticks       = 3
            yminor       = 0.05
            ysize        = numPix

        # pe,ped vs. Delta_pe

        if plotnumber == 4:

            outfilename = "pevsdelta"

            xquantity    = self.W_ped_psin_pe
            xlabel       = r'$\Delta_{\mathrm{ped,pe}}$'
            x1           = 0.0
            x2           = 0.1
            xticks       = 4
            xminor       = 0.025
            xsize        = numPix

            yquantity    = self.H_ped_psin_pe/1000.0
            ylabel       = r'$p_{\mathrm{e,ped}}$ (kPa)'
            y1           = 0.0
            y2           = 2
            yticks       = 4
            yminor       = 0.05
            ysize        = numPix

        # te,ped,r vs. W_r_te

        if plotnumber == 5:

            outfilename = "tevswr"

            xquantity    = self.W_ped_radius_te
            xlabel       = r'$W_{\mathrm{ped,Te}}$ (m)'
            x1           = 0.0
            x2           = 0.033
            xticks       = 3
            xminor       = 0.01
            xsize        = numPix

            yquantity    = self.H_ped_radius_te/1000.0
            ylabel       = r'$T_{\mathrm{e,ped,r}}$ (keV)'
            y1           = 0.0
            y2           = 0.4
            yticks       = 3
            yminor       = 0.05
            ysize        = numPix

        # ne,ped,r vs. W_r_ne

        if plotnumber == 6:

            outfilename = "nevswr"

            xquantity    = self.W_ped_radius_ne
            xlabel       = r'$W_{\mathrm{ped,ne}}$ (m)'
            x1           = 0.0
            x2           = 0.033
            xticks       = 3
            xminor       = 0.01
            xsize        = numPix

            yquantity    = self.H_ped_radius_ne/1.0e20
            ylabel       = r'$n_{\mathrm{e,ped,r}}$ ($10^{20}$ m$^{-3}$)'
            y1           = 0.0
            y2           = 0.6
            yticks       = 3
            yminor       = 0.05
            ysize        = numPix

        # pe,ped,r vs. W_r_pe

        if plotnumber == 7:

            outfilename = "pevswr"

            xquantity    = self.W_ped_radius_pe
            xlabel       = r'$W_{\mathrm{ped,pe}}$ (m)'
            x1           = 0.0
            x2           = 0.0333
            xticks       = 3
            xminor       = 0.01
            xsize        = numPix

            yquantity    = self.H_ped_radius_pe/1000.0
            ylabel       = r'$p_{\mathrm{e,ped,r}}$ (kPa)'
            y1           = 0.0
            y2           = 2
            yticks       = 3
            yminor       = 0.05
            ysize        = numPix
        # A vs kappa

        if plotnumber == 8:

            outfilename = "deltavskappa"

            xquantity    = self.elong
            xlabel       = r'$\kappa$'
            x1           = 1.8
            x2           = 2.2
            xticks       = 2
            xminor       = 0.1
            xsize        = numPix

            yquantity    = self.delta
            ylabel       = r'$\delta$'
            y1           = 0.4
            y2           = 0.6
            yticks       = 4
            yminor       = 0.05
            ysize        = numPix

        # A vs delta

        if plotnumber == 9:

            outfilename = "aratiovsdelta"

            xquantity    = self.aratio
            xlabel       = r'Aspect ratio'
            x1           = 1
            x2           = 3
            xticks       = 4
            xminor       = 0.1
            xsize        = numPix

            yquantity    = self.delta
            ylabel       = r'$\delta$ - Triangularity'
            y1           = -0.1
            y2           = 0.75
            yticks       = 4
            yminor       = 0.05
            ysize        = numPix
        
        #elong vs delta

        if plotnumber == 10:

            outfilename = "elongvsdelta"

            xquantity    = self.elong
            xlabel       = r'$\kappa$'
            x1           = 0
            x2           = 2.5
            xticks       = 4
            xminor       = 0.25
            xsize        = numPix

            yquantity    = self.delta
            ylabel       = r'$\delta$'
            y1           = 0
            y2           = 0.7
            yticks       = 4
            yminor       = 0.05
            ysize        = numPix

        #ne vs te
        if plotnumber == 11:
            outfilename = "nevste"
            xquantity    = self.H_ped_psin_ne/1.0e20
            xlabel       = r'$n_{\mathrm{e,ped}}$ ($10^{20}$ m$^{-3}$)'
            x1           = 0.0
            x2           = 0.6
            xticks       = 3
            xminor       = 0.05
            xsize        = numPix


            yquantity    = self.H_ped_psin_te/1000.0
            ylabel       = r'$T_{\mathrm{e,ped}}$ (keV)'
            y1           = 0.0
            y2           = 0.5
            yticks       = 5
            yminor       = 0.05
            ysize        = numPix

        # pressure cont vs temp cont
        if plotnumber == 12:
            outfilename = "FactorsVsDelta"
            yquantity    = (self.H_ped_psin_ne/1.0e20/self.W_ped_psin_ne*self.H_ped_psin_te/1000.0)/(self.H_ped_psin_ne/1.0e20*self.H_ped_psin_te/1000.0/self.W_ped_psin_te)
            ylabel       = r'1/2'
            y1           = -100
            y2           = 100
            yticks       = 3
            yminor       = 10
            ysize        = numPix


            xquantity    = self.W_ped
            xlabel       = r'$\Delta_{\mathrm{ped}}$'
            x1           = -1
            x2           = 1
            xticks       = 3
            xminor       = 0.025
            xsize        = numPix



        outfilename = str(self.shotNum) + outfilename + "Colored" + countType
        args = [plotnumber,outfilename,
                xquantity,xlabel,x1,x2,xticks,xminor,xsize,
                yquantity,ylabel,y1,y2,yticks,yminor,ysize, self]

        makefigure(*args)
    def makeAnimation(self, yvalue, saveAnim = True):
        """Creates an animation of thompson data vs radius

        Args:
            yvalue (str): Can select "te", "ne", "r", or "all".
            saveAnim (bool, optional): Saves animation to 'animations/{shotNum}{yvalue}.mp4'. Defaults to True.
        """
       
        if self.shotNum == "allShots":
            raise Exception("only run animation on single shot")
        if not self.client:
            raise Exception("Must have client data to run animation")
        def createAnimation(yvalue, yparam):
            '''Helper function which creates animation of yparam vs
            radius across time for shot number shotNum. yvalue is 
            the string representation of yparam.'''
            minimum = yparam.data[np.isfinite(yparam.data)].min()
            maximum = yparam.data[np.isfinite(yparam.data)].max()
            spread = maximum-minimum
            # initializing a figure in  
            # which the graph will be plotted 
            fig = plt.figure()  
            
            # marking the x-axis and y-axis 
            
            axis = plt.axes(xlim =(0.8, 1),  
                        ylim =(minimum - 0.1*spread, maximum + 0.1*spread),
                    xlabel="Radius (m?)",
                    ylabel = yvalue,
                    title = "Shot "+self.shotNum+": "+yvalue+" vs. Radius")
            
            # initializing a frame 
            pedPlot, = axis.plot([], [], ".")  
            
            # data which the line will  
            # contain (x, y) 
            def init():  
                pedPlot.set_data([], [])
                return pedPlot, 
            
            def animate(i): 
                x = np.linspace(0,1,130)
                if i<175:
                    print(str(i)+"/175", end="\r")
                if i==175:
                    print("Done    ", end = "\n")
                #update frame
                
                y = yparam.data[i,:]
                pedPlot.set_data(x, y) 
                return pedPlot, 

            anim = animation.FuncAnimation(fig, animate, init_func = init, 
                            frames = 175, interval = 20, blit = True,
                        repeat=False) 
            fig.canvas.draw()
            anim.event_source.stop()
            if saveAnim:
                anim.save('animations/'+self.shotNum+yvalue+'.mp4', writer = 'ffmpeg', fps = 22)
                print("Saved " + self.shotNum+yvalue+'.mp4 in animations')
        
        
        if yvalue == "all":
            createAnimation("te",self.te)
            createAnimation("r",self.r)
            createAnimation("ne",self.ne)
            return
        elif yvalue == "te":
            createAnimation("te",self.te)
        elif yvalue == "r":
            createAnimation("r",self.r)
        elif yvalue == "ne":
            createAnimation("ne",self.ne)


allShotNums = [47885, 47886, 47888, 47889, 47890, 47891, 47893, 47894, 47918, 47955, 47956, 47958, 47959, 47961, 47962, 47963, 47964, 47979, 47980, 47982, 47985, 47989, 47990, 47991, 47996, 47997, 47998, 47999, 48000, 48002, 48003, 48004, 48005, 48006, 48008, 48009, 48010, 48012, 48057, 48058, 48060, 48061, 48062, 48064, 48065, 48066, 48068, 48069, 48070, 48071, 48072, 48073, 48074, 48079, 48080, 48081, 48082, 48083, 48087, 48088, 48089, 48093, 48094, 48103, 48104, 48107, 48108, 48109, 48110, 48111, 48112, 48113, 48114, 48115, 48116, 48117, 48118, 48119, 48120, 48121, 48122, 48123, 48124, 48125, 48126, 48127, 48129, 48130, 48131, 48132, 48133, 48134, 48135, 48136, 48137, 48151, 48155, 48156, 48157, 48158, 48159, 48160, 48164, 48168, 48172, 48173, 48174, 48175, 48176, 48177, 48178, 48180, 48181, 48183, 48186, 48187, 48188, 48189, 48193, 48194, 48196, 48198, 48200, 48219, 48221, 48223, 48233, 48235, 48251, 48252, 48255, 48256, 48257, 48258, 48259, 48260, 48261, 48263, 48265, 48267, 48268, 48269, 48270, 48271, 48272, 48273, 48275, 48276, 48278, 48279, 48280, 48281, 48284, 48285, 48286, 48287, 48288, 48291, 48292, 48293, 48295, 48297, 48298, 48299, 48302, 48303, 48304, 48305, 48309, 48310, 48311, 48312, 48313, 48314, 48315, 48316, 48326, 48330, 48332, 48333, 48334, 48336, 48337, 48338, 48339, 48340, 48341, 48342, 48343, 48344, 48345, 48347, 48348, 48353, 48354, 48359, 48361, 48363, 48366, 48367, 48368, 48369, 48370, 48558, 48559, 48560, 48561, 48579, 48580, 48594, 48595, 48596, 48597, 48598, 48599, 48602, 48603, 48604, 48605, 48606, 48609, 48611, 48614, 48615, 48616, 48617, 48618, 48619, 48620, 48622, 48623, 48630, 48631, 48632, 48634, 48636, 48638, 48639, 48640, 48641, 48642, 48643, 48646, 48647, 48648, 48649, 48651, 48652, 48653, 48654, 48655, 48656, 48657, 48658, 48666, 48668, 48669, 48670, 48671, 48672, 48710, 48711, 48712, 48714, 48715, 48716, 48717, 48718, 48721, 48722, 48723, 48725, 48726, 48735, 48738, 48740, 48743, 48745, 48749, 48750, 48752, 48755, 48758, 48759, 48760, 48761, 48762, 48763, 48764, 48765, 48766, 48767, 48768, 48769, 48772, 48777, 48778, 48779, 48780, 48788, 48789, 48791, 48797, 48798, 48799, 48800, 48801, 48802, 48803, 48804, 48805, 48806, 48807, 48808, 48809, 48811, 48812, 48813, 48816, 48817, 48818, 48819, 48820, 48821, 48822, 48823, 48824, 48825, 48826, 48827, 48828, 48829, 48830, 48832, 48834, 48835, 48836, 48840, 48841, 48842, 48844, 48845, 48846, 48847, 48849, 48850, 48851, 48853, 48863, 48864, 48866, 48867, 48868, 48869, 48870, 48871, 48872, 48873, 48874, 48879, 48880, 48882, 48883, 48884, 48885, 48886, 48888, 48889, 48890, 48892, 48893, 48894, 48895, 48896, 48898, 48899, 48900, 48901, 48902, 48903, 48904, 48906, 48907, 48908, 48909, 48910, 48911, 48912, 48913, 48915, 48916, 48917, 48918, 48919, 48920, 48921, 48925, 48926, 48927, 48928, 48929, 48930, 48931, 48932, 48933, 48934, 48935, 48936, 49033, 49034, 49035, 49036, 49037, 49038, 49039, 49040, 49042, 49045, 49046, 49047, 49048, 49049, 49050, 49051, 49052, 49054, 49055, 49056, 49057, 49058, 49059, 49060, 49061, 49062, 49063, 49066, 49069, 49070, 49071, 49072, 49073, 49074, 49075, 49076, 49077, 49078, 49080, 49081, 49084, 49091, 49093, 49094, 49095, 49099, 49101, 49102, 49103, 49104, 49105, 49106, 49107, 49108, 49109, 49110, 49111, 49112, 49113, 49117, 49118, 49119, 49120, 49121, 49122, 49123, 49124, 49125, 49126, 49127, 49128, 49130, 49131, 49134, 49135, 49136, 49137, 49138, 49139, 49140, 49141, 49142, 49143, 49145, 49146, 49147, 49148, 49149, 49150, 49151, 49152, 49154, 49157, 49159, 49162, 49163, 49164, 49166, 49167, 49168, 49169, 49170, 49171, 49172, 49173, 49174, 49175, 49177, 49178, 49179, 49180, 49181, 49182, 49183, 49184, 49186, 49187, 49188, 49189, 49190, 49191, 49192, 49194, 49195, 49196, 49197, 49198, 49200, 49204, 49205, 49206, 49208, 49209, 49210, 49211, 49212, 49213, 49214, 49216, 49217, 49218, 49219, 49220, 49239, 49240, 49241, 49242, 49243, 49244, 49245, 49246, 49247, 49248, 49249, 49250, 49256, 49257, 49258, 49259, 49260, 49261, 49262, 49266, 49267, 49268, 49269, 49270, 49271, 49272, 49273, 49274, 49275, 49276, 49281, 49282, 49283, 49286, 49289, 49290, 49291, 49292, 49293, 49295, 49296, 49297, 49298, 49299, 49301, 49302, 49303, 49304, 49310, 49312, 49313, 49314, 49315, 49317, 49318, 49319, 49320, 49321, 49323, 49324, 49325, 49326, 49327, 49328, 49336, 49337, 49338, 49339, 49340, 49341, 49342, 49343, 49344, 49345, 49346, 49347, 49351, 49352, 49353, 49354, 49355, 49356, 49357, 49358, 49359, 49360, 49363, 49364, 49365, 49366, 49367, 49368, 49370, 49371, 49373, 49374, 49376, 49377, 49378, 49379, 49380, 49383, 49384, 49385, 49386, 49387, 49388, 49389, 49390, 49391, 49392, 49394, 49395, 49396, 49397, 49398, 49399, 49400, 49401, 49404, 49405, 49407, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49416, 49417, 49418, 49419, 49420, 49421, 49422, 49423, 49425, 49426, 49427, 49428, 49429, 49430, 49431, 49432, 49433, 49434, 49435, 49436, 49437, 49438, 49439, 49442, 49444, 49445, 49447, 49449, 49450, 49451, 49452, 49453, 49454, 49456, 49457, 49458, 49459, 49460, 49461, 49462, 49463, 49464, 49465, 49466, 49467, 49468]
# 47956,47982, 48071, 48156, 48291, 49063 , 49069 failed Ip
# 48137, 47885 49247, 49259, 49351, 49352, 49353, 49379, 49413, 49456failed nbi 
# [47956, 47982, 48071, 48156, 48291, 49063, 49069, 49247, 49259, 49351, 49352, 49353, 49379, 49413, 49456] failed ELMs
failedShotNums = [47885,47956, 47982, 48071, 48156, 48291, 49063, 49069, 49247, 49259, 49351, 49352, 49353, 49379, 49413, 49456]



# [47918, 47955, 47958, 47959, 47961, 47962, 47963, 47964, 47985, 47989, 47990, 47991, 47996, 47997, 47998, 47999, 48000, 48002, 48008, 48009, 48010, 48012, 48057, 48058, 48060, 48061, 48062, 48064, 48065, 48066, 48068, 48069, 48070, 48072, 48073, 48074, 48079, 48080, 48081, 48082, 48083, 48087, 48088, 48089, 48093, 48094, 48103, 48104, 48107, 48108, 48109, 48110, 48111, 48112, 48113, 48114, 48115, 48116, 48117, 48118, 48119, 48120, 48121, 48122, 48123, 48151, 48155, 48186, 48187, 48188, 48189, 48193, 48194, 48196, 48198, 48200, 48219, 48221, 48223, 48233, 48235, 48251, 48252, 48256, 48267, 48268, 48269, 48271, 48272, 48273, 48275, 48276, 48278, 48284, 48285, 48286, 48287, 48288, 48292, 48293, 48295, 48297, 48298, 48299, 48302, 48303, 48304, 48305, 48326, 48330, 48332, 48333, 48334, 48336, 48338, 48347, 48348, 48353, 48354, 48359, 48636, 48638, 48639, 48640, 48641, 48642, 48643, 48710, 48711, 48712, 48714, 48715, 48716, 48717, 48718, 48721, 48722, 48723, 48725, 48726, 48735, 48738, 48740, 48743, 48745, 48749, 48750, 48752, 48755, 48758, 48759, 48760, 48761, 48762, 48763, 48764, 48765, 48766, 48767, 48768, 48769, 48772, 48777, 48778, 48779, 48780, 48788, 48789, 48791, 48797, 48798, 48799, 48800, 48801, 48802, 48803, 48804, 48805, 48806, 48807, 48808, 48809, 48811, 48812, 48813, 48816, 48817, 48818, 48819, 48820, 48821, 48822, 48823, 48824, 48825, 48826, 48827, 48828, 48829, 48830, 48832, 48834, 48835, 48836, 48840, 48841, 48842, 48844, 48845, 48846, 48847, 48849, 48850, 48851, 48853, 48863, 48864, 48866, 48867, 48868, 48869, 48870, 48871, 48872, 48873, 48874, 48879, 48880, 48882, 48883, 48884, 48885, 48886, 48888, 48889, 48890, 48892, 48893, 48894, 48895, 48896, 48898, 48899, 48900, 48901, 48902, 48903, 48904, 48906, 48907, 48908, 48909, 48910, 48911, 48912, 48913, 48915, 48916, 48917, 48918, 48919, 48920, 48921, 48925, 48926, 48927, 48928, 48929, 48930, 48931, 48932, 48933, 48934, 48935, 48936, 49033, 49034, 49035, 49036, 49037, 49038, 49099, 49171, 49177, 49281, 49282, 49283, 49286, 49289, 49290, 49291, 49292, 49293, 49295, 49296, 49297, 49298, 49299, 49301, 49302, 49303, 49304, 49327, 49328, 49356, 49363, 49373, 49374, 49407, 49408, 49409, 49427, 49428, 49433, 49447, 49449, 49450, 49452, 49453, 49459]
# failed guns


# failed greenwald
#[47956, 47982, 48071, 48156, 48291, 49063, 49069, 49247, 49259, 49351, 49352, 49353, 49379, 49413, 49456]




def pickleCombine(shotNums = allShotNums, failedShotNums = failedShotNums, folder = "outputWithBeamPower3", outfileName = "MAST-U_pedestal_allShots"):
    """
    Combine pickle files

    J.W. Berkery 08/16/21: Initial version
    Edited C Fitzpatrick 20240618
    """

    #all shots
    infiles = []
    for i in shotNums:
        if i not in failedShotNums:
            infiles += [folder+'/MAST-U_pedestal_'+str(i)+'.pkl']
    outfile = open(folder+"/"+outfileName+".pkl", 'wb')

    infile_01      = open(infiles[0], 'rb')
    pkldata_01 = pickle.load(infile_01,encoding='latin1')
    infile_01.close()

    infile_02      = open(infiles[1], 'rb')
    pkldata_02 = pickle.load(infile_02,encoding='latin1')
    infile_02.close()

    keys = list(pkldata_01.keys())
    pkldata = {}
    pkldata = pkldata.fromkeys(keys)

    for k in pkldata:
        pkldata[k] = numpy.append(pkldata_01[k],pkldata_02[k])

    if len(infiles) > 2:
        for i in range(2,len(infiles)):
            infile_i      = open(infiles[i], 'rb')
            pkldata_i = pickle.load(infile_i,encoding='latin1')
            infile_i.close()

            for k in pkldata:
                pkldata[k] = numpy.append(pkldata[k],pkldata_i[k])


    pickle.dump(pkldata,outfile)

    outfile.close()

def importShots(shotNums = allShotNums, failedShotNums = failedShotNums):
    start_time = time.time()
    counter = 1
    totalNumShots = len(shotNums)-len(failedShotNums)
    failedShots = []
    for i in shotNums:
        if i not in failedShotNums:
            try:
                a = Shot(i, "client")
                # Adding new shots
                a.fit(savePklForShot=True)
                print("--- %s seconds ---" % (time.time() - start_time))
                print(str(totalNumShots-counter)+"left to go")
                print("\n")
            except Exception as error:
                failedShots += [i]
                print(error)
                print(i, " FAILED --------------------------------------")
        counter += 1
    print("failedShots = ", failedShots)
    failedShotNums += failedShots
    pickleCombine()












