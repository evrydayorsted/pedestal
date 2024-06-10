import matplotlib.pyplot as plt
import pickle
from archive.contourplot import *
from   matplotlib import rc
from   matplotlib.ticker import MultipleLocator
import copy
import matplotlib
import matplotlib.animation as animation
import numpy as np
try:
    import pyuda
except:
    print("pyuda connection failed")
from importlib import reload

class Shot:
    def __init__(self, shotNum, datatype):
        self.shotNum = str(shotNum)
        self.pkl = False
        self.client = False
        def pklDownload(self):
            try:
                #download pkl
                filename = 'output/MAST-U_pedestal_'+self.shotNum+'.pkl'
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
                print("Pkl data loaded")
            except:
                print("Pkl data procurement failed")
        def clientDownload(self,shotNum):
            print("Getting data from client for " +self.shotNum)
            try:
                client = pyuda.Client()
                self.te   = client.get('/ayc/t_e',shotNum)
                print("te downloaded")
                self.dte  = client.get('/ayc/dt_e',shotNum)
                print("dte downloaded")
                self.ne   = client.get('/ayc/n_e',shotNum)
                print("ne downloaded")
                self.dne  = client.get('/ayc/dn_e',shotNum)
                print("dne downloaded")
                self.r    = client.get('/ayc/r',shotNum)
                print("r downloaded")
                self.psinprof  = client.get('epm/output/radialprofiles/normalizedpoloidalflux',shotNum)
                print("psinprof downloaded")
                self.rprof     = client.get('epm/output/radialprofiles/R',shotNum)
                print("rprof downloaded")
                self.times_ayc = self.te.time.data
                self.client = True
                print("All data downloaded from client")
            except:
                print("Client connection failed.")
        if datatype == "pkl":
            pklDownload(self)
        elif datatype == "client":
            clientDownload(self, shotNum)
        elif datatype == "all":
            pklDownload(self)
            clientDownload(self, shotNum)
        else:
            raise Exception("datatype must be 'pkl,' 'client,' or 'all'")
    def __str__(self):
        return f"{self.shotNum}" 
    def contourPlot(self, plotnumber):
        '''1 for Beta vs. Delta\n
       2 for Te,ped vs. Delta_te\n
       3 for ne,ped vs. Delta_ne\n
       4 for pe,ped vs. Delta_pe\n
       5 for  te,ped,r vs. W_r_te\n
       6 for ne,ped,r vs. W_r_ne\n
       7 for pe,ped,r vs. W_r_pe'''
        if not self.pkl:
            raise Exception("Must have pkl data to run contourPlot")
        def setupfigure(figurenumber,xsize,ysize):
            figurename = plt.figure(figurenumber,figsize=(xsize,ysize),
                                    edgecolor='white')
            matplotlib.rcParams['xtick.major.pad'] = 8
            matplotlib.rcParams['ytick.major.pad'] = 8
            return figurename
        def setupframe(framecolumns,framerows,position,x1,x2,y1,y2,numberofxticks,
                    numberofyticks,xlabel,ylabel,xminor,yminor,font_size):
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

            figure1 = setupfigure(1,6.0,5.0)
            font_size = 20

            xx = np.linspace(x1,x2,num=xsize+1)
            yy = np.linspace(y1,y2,num=ysize+1)

            xx2 = np.delete(xx+(xx[1]-xx[0])/2.0,-1)
            yy2 = np.delete(yy+(yy[1]-yy[0])/2.0,-1)

            Ntot   = np.zeros((len(xx)-1,len(yy)-1))

            for i in range(0,len(xx)-1):
                for j in range(0,len(yy)-1):
                    index, = np.where((xquantity>=xx[i])   & 
                                        (xquantity< xx[i+1]) &
                                        (yquantity>=yy[j])   &
                                        (yquantity< yy[j+1]))
                    Ntot[i,j]   = len(index)


            zeroindex = np.where(Ntot == 0.0)
            Ntot[zeroindex] = 1.0e-10
            zz = np.transpose(np.log10(Ntot))

        #    print(zz)

            frame1 = setupframe(1,1,1,x1,x2,y1,y2,
                                xticks,yticks,xlabel,
                                ylabel,xminor,yminor,font_size)

            CS = plt.imshow(zz,extent=(x1,x2,y1,y2),origin='lower',
                            interpolation='none',
                            aspect='auto',
                            vmin=0.0,vmax=2.0)    
            cbar = plt.colorbar(CS,ticks=[0.0,1.0,2.0])#,3.0])
            cmap = copy.copy(matplotlib.colormaps.get_cmap('viridis'))
            cmap.set_under(color='white')
            cmap.set_bad(color='white')
            plt.set_cmap(cmap)
            cbar.ax.set_yticklabels(['0.0','1.0','2.0'],fontsize=font_size)#,'3.0'],fontsize=font_size)
            cbar.ax.set_ylabel('Log$_{10}$ (Number of equilibria)',fontsize=font_size)

            plt.subplots_adjust(left=0.20,right = 0.90,bottom=0.20,top=0.92)
            # This provides a square plot area with a 5in by 6in figure area and the colorbar on the right

            if plotnumber == 1:
                x_width = np.linspace(x1,x2,100)
                y_beta  = (x_width/0.43)**(1.0/1.03)
                y_beta2 = (x_width/0.08)**(2.0)
                plt.plot(x_width,y_beta,color='red',linestyle='--')
                plt.plot(x_width,y_beta2,color='magenta',linestyle='--')
                #plt.annotate(r'NSTX GCP: $\Delta_{\mathrm{ped}} = 0.43\beta_{\theta,\mathrm{ped}}^{1.03}$',(0.06,0.308),color='red',fontsize=13,annotation_clip=False)
                plt.annotate(r'$\Delta_{\mathrm{ped}} = 0.43\beta_{\theta,\mathrm{ped}}^{1.03}$',(0.12,y2+0.008),color='red',fontsize=13,annotation_clip=False)
                plt.annotate(r'$\Delta_{\mathrm{ped}} = 0.08\beta_{\theta,\mathrm{ped}}^{0.5}$',(0.0,y2+0.008),color='magenta',fontsize=13,annotation_clip=False)

            plt.savefig("plots/"+outfilename+'.pdf')
            plt.show()
        #    plt.close()

        # Beta vs. Delta
        
        if plotnumber == 1:

            outfilename = "betavsdelta"

            xquantity    = self.W_ped
            xlabel       = r'$\Delta_{\mathrm{ped}}$'
            x1           = 0.0
            x2           = 0.2
            xticks       = 4
            xminor       = 0.025
            xsize        = 60

            yquantity    = self.Beta_ped
            ylabel       = r'$\beta_{\theta,\mathrm{ped}}$'
            y1           = 0.0
            y2           = 0.35
            yticks       = 7
            yminor       = 0.025
            ysize        = 60

        # Te,ped vs. Delta_te

        if plotnumber == 2:

            outfilename = "tevsdelta"

            xquantity    = self.W_ped_psin_te
            xlabel       = r'$\Delta_{\mathrm{ped,Te}}$'
            x1           = 0.0
            x2           = 0.2
            xticks       = 4
            xminor       = 0.025
            xsize        = 60

            yquantity    = self.H_ped_psin_te/1000.0
            ylabel       = r'$T_{\mathrm{e,ped}}$ (keV)'
            y1           = 0.0
            y2           = 0.3
            yticks       = 3
            yminor       = 0.05
            ysize        = 60

        # ne,ped vs. Delta_ne

        if plotnumber == 3:

            outfilename = "nevsdelta"

            xquantity    = self.W_ped_psin_ne
            xlabel       = r'$\Delta_{\mathrm{ped,ne}}$'
            x1           = 0.0
            x2           = 0.2
            xticks       = 4
            xminor       = 0.025
            xsize        = 60

            yquantity    = self.H_ped_psin_ne/1.0e20
            ylabel       = r'$n_{\mathrm{e,ped}}$ ($10^{20}$ m$^{-3}$)'
            y1           = 0.0
            y2           = 0.6
            yticks       = 3
            yminor       = 0.05
            ysize        = 60

        # pe,ped vs. Delta_pe

        if plotnumber == 4:

            outfilename = "pevsdelta"

            xquantity    = self.W_ped_psin_pe
            xlabel       = r'$\Delta_{\mathrm{ped,pe}}$'
            x1           = 0.0
            x2           = 0.2
            xticks       = 4
            xminor       = 0.025
            xsize        = 60

            yquantity    = self.H_ped_psin_pe/1000.0
            ylabel       = r'$p_{\mathrm{e,ped}}$ (kPa)'
            y1           = 0.0
            y2           = 1.2
            yticks       = 4
            yminor       = 0.05
            ysize        = 60

        # te,ped,r vs. W_r_te

        if plotnumber == 5:

            outfilename = "tevswr"

            xquantity    = self.W_ped_radius_te
            xlabel       = r'$W_{\mathrm{ped,Te}}$ (m)'
            x1           = 0.0
            x2           = 0.09
            xticks       = 3
            xminor       = 0.01
            xsize        = 60

            yquantity    = self.H_ped_radius_te/1000.0
            ylabel       = r'$T_{\mathrm{e,ped,r}}$ (keV)'
            y1           = 0.0
            y2           = 0.3
            yticks       = 3
            yminor       = 0.05
            ysize        = 60

        # ne,ped,r vs. W_r_ne

        if plotnumber == 6:

            outfilename = "nevswr"

            xquantity    = sefl.W_ped_radius_ne
            xlabel       = r'$W_{\mathrm{ped,ne}}$ (m)'
            x1           = 0.0
            x2           = 0.09
            xticks       = 3
            xminor       = 0.01
            xsize        = 60

            yquantity    = self.H_ped_radius_ne/1.0e20
            ylabel       = r'$n_{\mathrm{e,ped,r}}$ ($10^{20}$ m$^{-3}$)'
            y1           = 0.0
            y2           = 0.6
            yticks       = 3
            yminor       = 0.05
            ysize        = 60

        # pe,ped,r vs. W_r_pe

        if plotnumber == 7:

            outfilename = "pevswr"

            xquantity    = self.W_ped_radius_pe
            xlabel       = r'$W_{\mathrm{ped,pe}}$ (m)'
            x1           = 0.0
            x2           = 0.09
            xticks       = 3
            xminor       = 0.01
            xsize        = 60

            yquantity    = self.H_ped_radius_pe/1000.0
            ylabel       = r'$p_{\mathrm{e,ped,r}}$ (kPa)'
            y1           = 0.0
            y2           = 1.2
            yticks       = 3
            yminor       = 0.05
            ysize        = 60
        outfilename = str(self.shotNum) + outfilename
        args = [plotnumber,outfilename,
                xquantity,xlabel,x1,x2,xticks,xminor,xsize,
                yquantity,ylabel,y1,y2,yticks,yminor,ysize]

        makefigure(*args)
    def makeAnimation(self, yvalue):
        '''Creates animation of yvalue ("te", "ne", "r", or "all") vs
        radius across time for shot number shotNum. Saves to files.'''
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
                if i<129:
                    print(str(i)+"/130", end="\r")
                if i==129:
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
        else:
            return