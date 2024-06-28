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

try:
    from pedinf.models import mtanh # type: ignore
except:
    print("pedinf connection failed")

try:
    import pyuda # type: ignore
except:
    print("pyuda connection failed")

class Shot:
    def __init__(self, shotNum, datatype):
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
            """Pulls data from pkl file 'output/MAST_U_pedestal_{#}.pkl' or 'output/MAST_U_pedestal_allShots.pkl'"""
            
            try:
                #download pkl
                filename = 'outputWithPlasmaCurrent/MAST-U_pedestal_'+self.shotNum+'.pkl'
                infile = open(filename, 'rb')
                pkldata = pickle.load(infile)
                infile.close()
                
                #read off values
                self.shot = pkldata['Shot']
                self.times = pkldata['Times']
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
                self.Aratio = pkldata['Aratio']
                self.elong  = pkldata['elong']
                self.delta = pkldata['delta']
                self.pkl = True
                try:
                    self.Ip = pkldata['Ip']
                except:
                    print("Plasma Current data failed")
                if self.shotNum == "allShots":
                    self.shotByTimeSlice = pkldata['ShotNum']
                    self.IpTime = pkldata['IpTime']
                    self.IpAdjusted = np.array(pkldata["IpTimeAdjusted"])
                    self.IpMax = np.array(pkldata['IpMax'])

                print("Pkl data loaded")
            except:
                print("Pkl data procurement failed")
        def clientDownload(self):
            '''Function to pull data from client. Should only be used for a single shot.'''
            print("Getting data from client for " +self.shotNum)
            try:
                client = pyuda.Client()

                # Defining shot based off of class instance
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
                print("done parameters")


                # EPM: EFIT++

                #neutral beam, Ip, toroidal magnetic field, stored energy, beta_N

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

                #self.psi_2D    = client.get('epm/output/profiles2D/poloidalflux',shot)
                print("All efit parameters loaded       ", end="\n")
                
                #in megawatts
                #self.total_NBI_power = client.get('anb/sum/power', shot)

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
                print("all ayc parameters loaded     ", end="\n")
                self.times_ayc = self.te.time.data
                self.client = True
                Ip = client.get('epm/output/globalParameters/plasmacurrent',self.shotNum)
                self.Ip = np.nan_to_num(Ip.data)
                
                self.IpTime = Ip.time.data
                print("All data downloaded from client")
            except:
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
        #total_NBI_power = self.total_NBI_power
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
                
        IpTime = self.IpTime
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

        # For each time slice
        for i in range(0,len(times)):
            
            time = times[i]
            
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
                ax1.set_ylim([0.,1.20*numpy.max(te_profile)])
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
                ax2.set_ylim([0,1.20*numpy.max(ne_profile/1e19)])
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

            pkldata = {'Shot': shot, 'Times': times, 'W_ped': W_ped, 'Beta_ped': beta_ped,
                    'W_ped_psin_te': W_ped_psin_te,'W_ped_psin_ne': W_ped_psin_ne,'W_ped_psin_pe': W_ped_psin_pe,
                    'H_ped_psin_te': H_ped_psin_te,'H_ped_psin_ne': H_ped_psin_ne,'H_ped_psin_pe': H_ped_psin_pe,
                    'W_ped_radius_te': W_ped_radius_te,'W_ped_radius_ne': W_ped_radius_ne,'W_ped_radius_pe': W_ped_radius_pe,
                    'H_ped_radius_te': H_ped_radius_te,'H_ped_radius_ne': H_ped_radius_ne,'H_ped_radius_pe': H_ped_radius_pe,
                    'Aratio': Aratio, 'elong': elong, 'delta': delta, 'Ip': self.Ip, 'IpTime': self.IpTime}
            filename = 'outputWithPlasmaCurrent/MAST-U_pedestal_'+str(shot)+'.pkl'
            outfile = open(filename, 'wb')
            pickle.dump(pkldata,outfile)
            outfile.close()



    def contourPlot(self, plotnumber, saveFigure=False, showFigure=True, fitHMode=False, plotName = "default", numPix = 60,
                    cbarMax =2, cbarMin=0, numMin = 10, countType = "count", IpMin = 0.9, lowSlopeFilter = True, posPed = True):
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
            # Counts number of points that lie within each bin
            for i in range(0,len(xx)-1):
                for j in range(0,len(yy)-1):
                    #Can add additional conditions here to filter what is shown in contour
                    if lowSlopeFilter:
                        if posPed:
                            index, = np.where((xquantity>=xx[i])   & 
                                                (xquantity< xx[i+1]) &
                                                (yquantity>=yy[j])   &
                                                (yquantity< yy[j+1]) &
                                                (self.IpAdjusted>IpMin*self.IpMax) &
                                                (self.Beta_ped/self.W_ped >0.75) &
                                                (self.H_ped_psin_ne>0) &
                                                (self.H_ped_psin_pe>0) )
                        else:
                            index, = np.where((xquantity>=xx[i])   & 
                                                (xquantity< xx[i+1]) &
                                                (yquantity>=yy[j])   &
                                                (yquantity< yy[j+1]) &
                                                (self.IpAdjusted>IpMin*self.IpMax) &
                                                (self.Beta_ped/self.W_ped >0.75))
                    elif posPed:
                        index, = np.where((xquantity>=xx[i])   & 
                                                (xquantity< xx[i+1]) &
                                                (yquantity>=yy[j])   &
                                                (yquantity< yy[j+1]) &  
                                                (self.IpAdjusted>IpMin*self.IpMax) &
                                                (self.H_ped_psin_ne>0) &
                                                (self.H_ped_psin_pe>0))
                    else:
                        index, = np.where((xquantity>=xx[i])   & 
                                                (xquantity< xx[i+1]) &
                                                (yquantity>=yy[j])   &
                                                (self.IpAdjusted>IpMin*self.IpMax) &
                                                (yquantity< yy[j+1]))  
                    if len(index) >= numMin:
                        totalPoints += len(index)

                        if countType == "count":
                            if len(index) == 0:
                                Ntot[i,j] = -1e-10
                            else:
                                Ntot[i,j] = np.log10(len(index))
                        elif countType == 'elong':
                            Ntot[i,j]   = np.mean(self.elong[index])
                        elif countType == "delta":
                            Ntot[i,j] = np.mean(self.delta[index])
                        elif countType == "time":
                            Ntot[i,j] = np.mean(self.times[index])
                        elif countType == "aratio":
                            Ntot[i,j] = np.mean(self.Aratio[index])
                        elif countType == "pedestalHeight":
                            Ntot[i,j] = np.mean(self.Beta_ped[index])
                        elif countType == "pedestalSlope":
                            Ntot[i,j] = np.mean(self.Beta_ped[index]/self.W_ped[index])


                    else:
                        Ntot[i,j] = None
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
                cbar.ax.set_ylabel(r'Avg. $\kappa$',fontsize=font_size)
            elif countType == "delta":
                cbar.ax.set_ylabel(r'Avg. $\delta$',fontsize=font_size)
            elif countType == "time":
                cbar.ax.set_ylabel(r'Avg. Time',fontsize=font_size)
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
            x1           = -0.1
            x2           = 1
            xticks       = 3
            xminor       = 0.025
            xsize        = numPix

            yquantity    = self.Beta_ped
            ylabel       = r'$\beta_{\theta,\mathrm{ped}}$'
            y1           = -0.1
            y2           =1
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
            x2           = 0.2
            xticks       = 4
            xminor       = 0.025
            xsize        = numPix

            yquantity    = self.H_ped_psin_pe/1000.0
            ylabel       = r'$p_{\mathrm{e,ped}}$ (kPa)'
            y1           = 0.0
            y2           = 1.2
            yticks       = 4
            yminor       = 0.05
            ysize        = numPix

        # te,ped,r vs. W_r_te

        if plotnumber == 5:

            outfilename = "tevswr"

            xquantity    = self.W_ped_radius_te
            xlabel       = r'$W_{\mathrm{ped,Te}}$ (m)'
            x1           = 0.0
            x2           = 0.09
            xticks       = 3
            xminor       = 0.01
            xsize        = numPix

            yquantity    = self.H_ped_radius_te/1000.0
            ylabel       = r'$T_{\mathrm{e,ped,r}}$ (keV)'
            y1           = 0.0
            y2           = 0.3
            yticks       = 3
            yminor       = 0.05
            ysize        = numPix

        # ne,ped,r vs. W_r_ne

        if plotnumber == 6:

            outfilename = "nevswr"

            xquantity    = self.W_ped_radius_ne
            xlabel       = r'$W_{\mathrm{ped,ne}}$ (m)'
            x1           = 0.0
            x2           = 0.09
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
            x2           = 0.09
            xticks       = 3
            xminor       = 0.01
            xsize        = numPix

            yquantity    = self.H_ped_radius_pe/1000.0
            ylabel       = r'$p_{\mathrm{e,ped,r}}$ (kPa)'
            y1           = 0.0
            y2           = 1.2
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

            xquantity    = self.Aratio
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
            
            axis = plt.axes(xlim =(0, 1.2),  
                        ylim =(minimum - 0.1*spread, maximum + 0.1*spread),
                    xlabel="Radius (m?)",
                    ylabel = yvalue,
                    title = "Shot "+self.shotNum+": "+yvalue+" vs. Radius")
            
            # initializing a frame 
            pedPlot, = axis.plot([], [], lw = 3)  
            
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

