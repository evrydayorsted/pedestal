import numpy
import matplotlib.pyplot as plt
import pickle
from contourplot import *
from   matplotlib import rc
from   matplotlib.ticker import MultipleLocator
import copy

def loadpkl(shotNum):
    #download pkl
    filename = 'output/MAST-U_pedestal_'+str(shotNum)+'.pkl'
    infile = open(filename, 'rb')
    pkldata = pickle.load(infile)
    infile.close()
    
    #initialize global variables
    global shot
    global times
    global W_ped
    global Beta_ped 
    global W_ped_psin_te
    global W_ped_psin_ne 
    global W_ped_psin_pe
    global H_ped_psin_te
    global H_ped_psin_ne
    global H_ped_psin_pe
    global W_ped_radius_te
    global W_ped_radius_ne
    global W_ped_radius_pe
    global H_ped_radius_te
    global H_ped_radius_ne
    global H_ped_radius_pe 
    global Aratio
    global elong 
    global delta

    #read off values
    shot = pkldata['Shot']
    times = pkldata['Times']
    W_ped = pkldata['W_ped']
    Beta_ped = pkldata['Beta_ped']
    W_ped_psin_te = pkldata['W_ped_psin_te']
    W_ped_psin_ne = pkldata['W_ped_psin_ne']
    W_ped_psin_pe = pkldata['W_ped_psin_pe']
    H_ped_psin_te = pkldata['H_ped_psin_te']
    H_ped_psin_ne = pkldata['H_ped_psin_ne']
    H_ped_psin_pe = pkldata['H_ped_psin_pe']
    W_ped_radius_te = pkldata['W_ped_radius_te']
    W_ped_radius_ne = pkldata['W_ped_radius_ne']
    W_ped_radius_pe = pkldata['W_ped_radius_pe']
    H_ped_radius_te = pkldata['H_ped_radius_te']
    H_ped_radius_ne = pkldata['H_ped_radius_ne']
    H_ped_radius_pe = pkldata['H_ped_radius_pe']
    Aratio = pkldata['Aratio']
    elong  = pkldata['elong']
    delta = pkldata['delta']

loadpkl(48136)

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

    xx = numpy.linspace(x1,x2,num=xsize+1)
    yy = numpy.linspace(y1,y2,num=ysize+1)

    xx2 = numpy.delete(xx+(xx[1]-xx[0])/2.0,-1)
    yy2 = numpy.delete(yy+(yy[1]-yy[0])/2.0,-1)

    Ntot   = numpy.zeros((len(xx)-1,len(yy)-1))

    for i in range(0,len(xx)-1):
        for j in range(0,len(yy)-1):
            index, = numpy.where((xquantity>=xx[i])   & 
                                 (xquantity< xx[i+1]) &
                                 (yquantity>=yy[j])   &
                                 (yquantity< yy[j+1]))
            Ntot[i,j]   = len(index)

    print(numpy.sum(Ntot))

    zeroindex = numpy.where(Ntot == 0.0)
    Ntot[zeroindex] = 1.0e-10
    zz = numpy.transpose(numpy.log10(Ntot))

#    print(zz)

    frame1 = setupframe(1,1,1,x1,x2,y1,y2,
                        xticks,yticks,xlabel,
                        ylabel,xminor,yminor,font_size)

    CS = plt.imshow(zz,extent=(x1,x2,y1,y2),origin='lower',
                    interpolation='none',
                    aspect='auto',
                    vmin=0.0,vmax=2.0)    
    cbar = plt.colorbar(CS,ticks=[0.0,1.0,2.0])#,3.0])
    cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
    cmap.set_under(color='white')
    cmap.set_bad(color='white')
    plt.set_cmap(cmap)
    cbar.ax.set_yticklabels(['0.0','1.0','2.0'],fontsize=font_size)#,'3.0'],fontsize=font_size)
    cbar.ax.set_ylabel('Log$_{10}$ (Number of equilibria)',fontsize=font_size)

    plt.subplots_adjust(left=0.20,right = 0.90,bottom=0.20,top=0.92)
    # This provides a square plot area with a 5in by 6in figure area and the colorbar on the right

    if plotnumber == 1:
        x_width = numpy.linspace(x1,x2,100)
        y_beta  = (x_width/0.43)**(1.0/1.03)
        y_beta2 = (x_width/0.08)**(2.0)
        plt.plot(x_width,y_beta,color='red',linestyle='--')
        plt.plot(x_width,y_beta2,color='magenta',linestyle='--')
        #plt.annotate(r'NSTX GCP: $\Delta_{\mathrm{ped}} = 0.43\beta_{\theta,\mathrm{ped}}^{1.03}$',(0.06,0.308),color='red',fontsize=13,annotation_clip=False)
        plt.annotate(r'$\Delta_{\mathrm{ped}} = 0.43\beta_{\theta,\mathrm{ped}}^{1.03}$',(0.12,y2+0.008),color='red',fontsize=13,annotation_clip=False)
        plt.annotate(r'$\Delta_{\mathrm{ped}} = 0.08\beta_{\theta,\mathrm{ped}}^{0.5}$',(0.0,y2+0.008),color='magenta',fontsize=13,annotation_clip=False)

    plt.savefig(outfilename+'.pdf')
    plt.show()
#    plt.close()




def contourPlot(plotnumber, shotNum):
    # Beta vs. Delta
    #download pkl
    filename = 'output/MAST-U_pedestal_'+str(shotNum)+'.pkl'
    infile = open(filename, 'rb')
    pkldata = pickle.load(infile)
    infile.close()
    global outfilename
    global xquantity
    global xlabel
    global x1
    global x2
    global xticks
    global xminor
    global xsize

    if plotnumber == 1:

        outfilename = "betavsdelta"

        xquantity    = pkldata['W_ped']
        xlabel       = r'$\Delta_{\mathrm{ped}}$'
        x1           = 0.0
        x2           = 0.2
        xticks       = 4
        xminor       = 0.025
        xsize        = 60

        yquantity    = pkldata['Beta_ped']
        ylabel       = r'$\beta_{\theta,\mathrm{ped}}$'
        y1           = 0.0
        y2           = 0.35
        yticks       = 7
        yminor       = 0.025
        ysize        = 60

    # Te,ped vs. Delta_te

    if plotnumber == 2:

        outfilename = "tevsdelta"

        xquantity    = pkldata['W_ped_psin_te']
        xlabel       = r'$\Delta_{\mathrm{ped,Te}}$'
        x1           = 0.0
        x2           = 0.2
        xticks       = 4
        xminor       = 0.025
        xsize        = 60

        yquantity    = pkldata['H_ped_psin_te']/1000.0
        ylabel       = r'$T_{\mathrm{e,ped}}$ (keV)'
        y1           = 0.0
        y2           = 0.3
        yticks       = 3
        yminor       = 0.05
        ysize        = 60

    # ne,ped vs. Delta_ne

    if plotnumber == 3:

        outfilename = "nevsdelta"

        xquantity    = pkldata['W_ped_psin_ne']
        xlabel       = r'$\Delta_{\mathrm{ped,ne}}$'
        x1           = 0.0
        x2           = 0.2
        xticks       = 4
        xminor       = 0.025
        xsize        = 60

        yquantity    = pkldata['H_ped_psin_ne']/1.0e20
        ylabel       = r'$n_{\mathrm{e,ped}}$ ($10^{20}$ m$^{-3}$)'
        y1           = 0.0
        y2           = 0.6
        yticks       = 3
        yminor       = 0.05
        ysize        = 60

    # pe,ped vs. Delta_pe

    if plotnumber == 4:

        outfilename = "pevsdelta"

        xquantity    = pkldata['W_ped_psin_pe']
        xlabel       = r'$\Delta_{\mathrm{ped,pe}}$'
        x1           = 0.0
        x2           = 0.2
        xticks       = 4
        xminor       = 0.025
        xsize        = 60

        yquantity    = pkldata['H_ped_psin_pe']/1000.0
        ylabel       = r'$p_{\mathrm{e,ped}}$ (kPa)'
        y1           = 0.0
        y2           = 1.2
        yticks       = 4
        yminor       = 0.05
        ysize        = 60

    # te,ped,r vs. W_r_te

    if plotnumber == 5:

        outfilename = "tevswr"

        xquantity    = pkldata['W_ped_radius_te']
        xlabel       = r'$W_{\mathrm{ped,Te}}$ (m)'
        x1           = 0.0
        x2           = 0.09
        xticks       = 3
        xminor       = 0.01
        xsize        = 60

        yquantity    = pkldata['H_ped_radius_te']/1000.0
        ylabel       = r'$T_{\mathrm{e,ped,r}}$ (keV)'
        y1           = 0.0
        y2           = 0.3
        yticks       = 3
        yminor       = 0.05
        ysize        = 60

    # ne,ped,r vs. W_r_ne

    if plotnumber == 6:

        outfilename = "nevswr"

        xquantity    = pkldata['W_ped_radius_ne']
        xlabel       = r'$W_{\mathrm{ped,ne}}$ (m)'
        x1           = 0.0
        x2           = 0.09
        xticks       = 3
        xminor       = 0.01
        xsize        = 60

        yquantity    = pkldata['H_ped_radius_ne']/1.0e20
        ylabel       = r'$n_{\mathrm{e,ped,r}}$ ($10^{20}$ m$^{-3}$)'
        y1           = 0.0
        y2           = 0.6
        yticks       = 3
        yminor       = 0.05
        ysize        = 60

    # pe,ped,r vs. W_r_pe

    if plotnumber == 7:

        outfilename = "pevswr"

        xquantity    = pkldata['W_ped_radius_pe']
        xlabel       = r'$W_{\mathrm{ped,pe}}$ (m)'
        x1           = 0.0
        x2           = 0.09
        xticks       = 3
        xminor       = 0.01
        xsize        = 60

        yquantity    = pkldata['H_ped_radius_pe']/1000.0
        ylabel       = r'$p_{\mathrm{e,ped,r}}$ (kPa)'
        y1           = 0.0
        y2           = 1.2
        yticks       = 3
        yminor       = 0.05
        ysize        = 60
    outfilename = "shotNum" + outfilename
    args = [plotnumber,outfilename,
            xquantity,xlabel,x1,x2,xticks,xminor,xsize,
            yquantity,ylabel,y1,y2,yticks,yminor,ysize]

    makefigure(*args)










