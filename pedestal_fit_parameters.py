from numpy import array, linspace
import matplotlib.pyplot as plt
import pyuda
import numpy
import scipy
import pickle
from pedinf.models import mtanh


shots = [
#47885,47886,47888,47889,47890,47891,47893,47894
#47918,47955,47956,47958,47959,47961,47962,47963,47964,47979,47980,47982,47985,47989,47990,47991,47996,47997,47998,47999,48000,48002,48003,48004,48005,48006,48008, 48009, 48010, 48012, 48057, 48058, 48060, 48061, 48062, 48064, 48065, 48066, 48068, 48069, 48070, 48071, 48072, 48073, 48074, 48079, 48080, 48081, 48082, 48083, 48087, 48088, 48089, 48093, 48094,48103,48104,48107,48108,48109,48110,48111,48112,48113,48114,48115,48116,48117,48118,48119,48120,48121,48122,48123,48124,48125,48126,48127,48129,48130,48131,48132,48133,48134,48135,48136,48137,48151,48155,48156,48157,48158,48159,48160,48164,48168,48172,48173,48174,48175,48176,48177,48178,48180,48181,48183,48186,48187,48188,48189,48193,48194,48196,48198,48200,48219,48221,48223,48233,48235,48251,48252,48255,48256,48257,48258,48259,48260,48261,48263,48265,48267,48268,48269,48270,48271,48272,48273,48275,48276,48278,48279,48280,48281,48284,48285,48286,48287,48288,48291,48292,48293,48295,48297,48298,48299,48302,48303,48304,48305,48309,48310,48311,48312,48313,48314,48315,48316,48326,48330,48332,48333,48334,48336,48337,48338,48339,48340,48341,48342,48343,48344,48345,48347,48348,48353,48354,48359,48361,48363,48366,48367,48368,48369,48370,48558,48559,48560,48561,48579,48580,48594,48595,48596,48597,48598,48599,48602,48603,48604,48605,48606,48609,48611,48614,48615,48616,48617,48618,48619,48620,48622,48623,48630,48631,48632,48634,48636,48638,48639,48640,48641,48642,48643,48646,48647,48648,48649,48651,48652,48653,48654,48655,48656,48657,48658,48666,48668,48669,48670,48671,48672,48710,48711,48712,48714,48715,48716,48717,48718,48721,48722,48723,48725,48726
#48735,48738,48740,48743,48745,48749,48750,48752,48755,48758,48759,48760,48761,48762,48763,48764,48765,48766,48767,48768,48769,48772,48777,48778,48779,48780,48788,48789,48791,48797,48798,48799,48800,48801,48802,48803,48804,48805,48806,48807,48808,48809,48811,48812,48813,48816,48817,48818,48819,48820,48821,48822,48823,48824,48825,48826,48827,48828,48829,48830,48832,48834,48835,48836,48840,48841,48842,48844,48845,48846,48847,48849,48850,48851,48853,48863,48864,48866,48867,48868,48869,48870,48871,48872,48873,48874,48879,48880,48882,48883,48884,48885,48886,48888,48889,48890,48892,48893,48894,48895,48896,48898,48899,48900,48901,48902,48903,48904,48906,48907,48908,48909,48910,48911,48912,48913,48915,48916,48917,48918,48919,48920,48921,48925,48926,48927,48928,48929,48930,48931,48932,48933,48934,48935,48936,49033,49034,49035,49036,49037,49038,49039,49040,49042,49045,49046,49047,49048,49049,49050,49051,49052,49054,49055,49056,49057,49058,49059,49060,49061,49062,49063,49066,49069,49070,49071,49072,49073,49074,49075,49076,49077,49078,49080,49081,49084,49091,49093,49094,49095,49099,49101,49102,49103,49104,49105,49106,49107,49108,49109,49110,49111,49112,49113,49117,49118,49119,49120,49121,49122,49123,49124,49125,49126,49127,49128,49130,49131,49134,49135,49136,49137,49138,49139,49140,49141,49142,49143,49145,49146,49147,49148,49149,49150,49151,49152,49154,49157,49159,49162,49163,49164,49166,49167,49168,49169,49170,49171,49172,49173,49174,49175,49177,49178,49179,49180,49181,49182,49183,49184,49186,49187,49188,49189,49190,49191,49192,49194,49195,49196,49197,49198,49200,49204,49205,49206,49208,49209,49210,49211,49212,49213,49214,49216,49217,49218,49219,49220
]

shots = [49220]


printtimes      = False
plotvstime      = False
printquantities = False
plotvsradius    = False
plotvspsin      = True
savepklforshot  = False

client = pyuda.Client()


for shot in shots:

    print("")
    print(shot)
    print("")
 
    # APF: automatic profile fitting

    group = "/apf/core/mtanh/lfs/"

    te_ped_location = client.get(group + "t_e/pedestal_location", shot)
    te_ped_height   = client.get(group + "t_e/pedestal_height", shot)
    te_ped_width    = client.get(group + "t_e/pedestal_width", shot)
    te_ped_top_grad = client.get(group + "t_e/pedestal_top_gradient", shot)
    te_background   = client.get(group + "t_e/background_level", shot)

    times_apf       = te_ped_location.time.data

    ne_ped_location = client.get(group + "n_e/pedestal_location", shot)
    ne_ped_height   = client.get(group + "n_e/pedestal_height", shot)
    ne_ped_width    = client.get(group + "n_e/pedestal_width", shot)
    ne_ped_top_grad = client.get(group + "n_e/pedestal_top_gradient", shot)
    ne_background   = client.get(group + "n_e/background_level", shot)

    pe_ped_location = client.get(group + "p_e/pedestal_location", shot)
    pe_ped_height   = client.get(group + "p_e/pedestal_height", shot)
    pe_ped_width    = client.get(group + "p_e/pedestal_width", shot)
    pe_ped_top_grad = client.get(group + "p_e/pedestal_top_gradient", shot)
    pe_background   = client.get(group + "p_e/background_level", shot)
    print("done parameters")
    # group the temperature and density parameters together so it's easy
    # to pass them into the model:
    te_parameters = array([
        te_ped_location.data,
        te_ped_height.data,
        te_ped_width.data,
        te_ped_top_grad.data,
        te_background.data
    ])

    ne_parameters = array([
        ne_ped_location.data,
        ne_ped_height.data,
        ne_ped_width.data,
        ne_ped_top_grad.data,
        ne_background.data
    ])

    pe_parameters = array([
        pe_ped_location.data,
        pe_ped_height.data,
        pe_ped_width.data,
        pe_ped_top_grad.data,
        pe_background.data
    ])
    
    # AYC: Thomson data

    if plotvsradius:
        te   = client.get('/ayc/t_e',shot)
        dte  = client.get('/ayc/dt_e',shot)
        ne   = client.get('/ayc/n_e',shot)
        dne  = client.get('/ayc/dn_e',shot)
        r    = client.get('/ayc/r',shot)
    
        times_ayc = te.time.data
    print('done thomson')
    # EPM: EFIT++

    Ip        = client.get('epm/output/globalParameters/plasmacurrent',shot)
    times_epm = Ip.time.data
    Btor      = client.get('epm/output/globalParameters/bphirmag',shot)
    rmaxis    = client.get('epm/output/globalParameters/magneticAxis/R',shot)
    zmaxis    = client.get('epm/output/globalParameters/magneticAxis/Z',shot)
    rbdy      = client.get('epm/output/separatrixGeometry/rboundary',shot)
    zbdy      = client.get('epm/output/separatrixGeometry/zboundary',shot)
    rmidin    = client.get('epm/output/separatrixGeometry/rmidplaneIn',shot)
    rmidout   = client.get('epm/output/separatrixGeometry/rmidplaneOut',shot)
    aminor    = client.get('epm/output/separatrixGeometry/minorRadius',shot)
    kappa     = client.get('epm/output/separatrixGeometry/elongation',shot)
    deltaup   = client.get('epm/output/separatrixGeometry/upperTriangularity',shot)
    deltalow  = client.get('epm/output/separatrixGeometry/lowerTriangularity',shot)
    #pmaxis    = client.get('epm/output/globalParameters/psiAxis',shot)
    #psibdy    = client.get('epm/output/globalParameters/psiBoundary',shot)

    rprof     = client.get('epm/output/radialprofiles/R',shot)
    psinprof  = client.get('epm/output/radialprofiles/normalizedpoloidalflux',shot)

    r_2D      = client.get('epm/output/profiles2D/R',shot)
    z_2D      = client.get('epm/output/profiles2D/Z',shot)
    psin_2D   = client.get('epm/output/profiles2D/psinorm',shot)
    psi_2D    = client.get('epm/output/profiles2D/poloidalflux',shot)
    print('done efit')
    ultimatemintime = 0.1
    mintime   = numpy.max([numpy.min(times_apf),numpy.min(times_epm),ultimatemintime])
    maxtime   = numpy.min([numpy.max(times_apf),numpy.max(times_epm)])
    time_index = numpy.where((times_apf >= mintime) & (times_apf <= maxtime))[0]
    times0    = numpy.array(times_apf[time_index])


    # First check if the rprof data from epm is good (at least two points > rmaxis exist). This filters out if rprof is all nans
    
    times = []
    
    for j in range(0,len(times0)):

        test_index_apf = numpy.argmin(abs(times_apf-times0[j]))
        test_index_epm = numpy.argmin(abs(times_epm-times_apf[test_index_apf]))
        index          = numpy.where(rprof.data[test_index_epm]>rmaxis.data[test_index_epm])[0]
        if len(index)  > 2:
            times.append(times0[j])


#    times = [0.173,0.456]
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

    for i in range(0,len(times)):
        
        time = times[i]
        
        time_index_apf = numpy.argmin(abs(times_apf-time))
        if plotvsradius:
            time_index_ayc = numpy.argmin(abs(times_ayc-times_apf[time_index_apf]))
        time_index_epm = numpy.argmin(abs(times_epm-times_apf[time_index_apf]))

        if printtimes:
            print("Time = ",times_apf[time_index_apf])
            #if plotvsradius:
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
        radius = linspace(r0, r1, npnts)

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


        if printquantities:

            print("")
            #print("Psin_mid = ",psin_mid)
            #print("Psin_ped = ",psin_ped_top)
            #print("pped     = ",pped)
            #print("Time     = ",times_apf[time_index_apf])
            print("W_ped    = ",wped_psin)
            print("beta_ped = ",beta)
            print("")

        fs = 16

        if plotvsradius:
        
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
            ax2.set_ylim([0.,1.20*numpy.max(ne_profile/1e19)])
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
            plt.show()


        if plotvspsin:
        
            # Plot the profiles at a given time, vs. psin

            fig = plt.figure(figsize=(6,10))
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            fig.suptitle(f"{shot} @ {te_ped_location.time.data[time_index_apf]:.3f} ms",fontsize=fs)

            ax1.plot(psin, te_profile[index2], lw=2, color="red")
            ymax = 1.20*numpy.max(te_profile)
            #ax1.plot((psin_ped_te,psin_ped_te),(0.0,teped), lw=2, color='black', linestyle='--')
            #ax1.plot((psin_ped_te_top,psin_ped_te_top),(0.0,teped), lw=2, color='black', linestyle=':')
            #ax1.plot((psin_ped_te_bot,psin_ped_te_bot),(0.0,teped), lw=2, color='black', linestyle=':')
            ax1.plot((psin_mid,psin_mid),(0.0,ymax), lw=2, color='blue', linestyle='--')
            ax1.plot((psin_ped_top,psin_ped_top),(0.0,ymax), lw=2, color='blue', linestyle=':')
            ax1.plot((psin_ped_bot,psin_ped_bot),(0.0,ymax), lw=2, color='blue', linestyle=':')
            ax1.set_xlabel("")
            ax1.set_ylabel("$T_{e}$ (eV)",fontsize=fs)
            #ax1.set_ylim([0.0, ymax])
            ax1.set_ylim([0.0, 200.0])
            #ax1.set_xlim([0.9, 1.05])
            ax1.set_xlim([0.85, 1.05])
            ax1.tick_params(axis='x',labelsize=fs)
            ax1.tick_params(axis='y',labelsize=fs)
            ax1.tick_params(labelbottom=False)

            ax2.plot(psin, ne_profile[index2]/1e19, lw=2, color="red")
            ymax = 1.20*numpy.max(ne_profile/1e19)
            #ax2.plot((psin_ped_ne,psin_ped_ne),(0.0,neped), lw=2, color='black', linestyle='--')
            #ax2.plot((psin_ped_ne_top,psin_ped_ne_top),(0.0,neped), lw=2, color='black', linestyle=':')
            #ax2.plot((psin_ped_ne_bot,psin_ped_ne_bot),(0.0,neped), lw=2, color='black', linestyle=':')
            ax2.plot((psin_mid,psin_mid),(0.0,ymax), lw=2, color='blue', linestyle='--')
            ax2.plot((psin_ped_top,psin_ped_top),(0.0,ymax), lw=2, color='blue', linestyle=':')
            ax2.plot((psin_ped_bot,psin_ped_bot),(0.0,ymax), lw=2, color='blue', linestyle=':')
            ax2.set_ylabel("$n_{e}$ ($10^{19}$ m$^{-3}$)",fontsize=fs)
            #ax2.set_ylim([0.0, ymax])
            ax2.set_ylim([0.0, 6.0])
            #ax2.set_xlim([0.9, 1.05])
            ax2.set_xlim([0.85, 1.05])
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
            ax3.set_xlim([0.85, 1.05])
            ax3.tick_params(axis='x',labelsize=fs)
            ax3.tick_params(axis='y',labelsize=fs)

            plt.tight_layout()
            plt.show()

            
    if plotvstime:

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

    
    if savepklforshot:

        pkldata = {'Shot': shot, 'Times': times, 'W_ped': W_ped, 'Beta_ped': beta_ped,
                   'W_ped_psin_te': W_ped_psin_te,'W_ped_psin_ne': W_ped_psin_ne,'W_ped_psin_pe': W_ped_psin_pe,
                   'H_ped_psin_te': H_ped_psin_te,'H_ped_psin_ne': H_ped_psin_ne,'H_ped_psin_pe': H_ped_psin_pe,
                   'W_ped_radius_te': W_ped_radius_te,'W_ped_radius_ne': W_ped_radius_ne,'W_ped_radius_pe': W_ped_radius_pe,
                   'H_ped_radius_te': H_ped_radius_te,'H_ped_radius_ne': H_ped_radius_ne,'H_ped_radius_pe': H_ped_radius_pe,
                   'Aratio': Aratio, 'elong': elong, 'delta': delta}
        filename = 'output/MAST-U_pedestal_'+str(shot)+'.pkl'
        outfile = open(filename, 'wb')
        pickle.dump(pkldata,outfile)
        outfile.close()

