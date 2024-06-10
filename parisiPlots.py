import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyuda
import pickle
from importlib import reload


def getClientData(shot):
	print("getting data for " +str(shot))
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
	print("psinprof downloaded")
	rprof     = client.get('epm/output/radialprofiles/R',shot)
	print("rprof downloaded")
	times_ayc = te.time.data
	print("All Data Downloaded")
	#did this to see relationshiop between psinprof and rprof
	#return (psinprof, rprof)
	return (te,dte,ne,dne,r, rprof, psinprof,times_ayc)

def makeAnimation(yvalue, shotNum):
	'''Creates animation of yvalue ("te", "ne", "r", or "all") vs
	radius across time for shot number shotNum. Saves to files.'''
	def createAnimation(yvalue, yparam, shotNum):
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
				title = "Shot "+str(shotNum)+": "+yvalue+" vs. Radius")
		
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
		
		anim.save('animations/'+str(shotNum)+yvalue+'.mp4', writer = 'ffmpeg', fps = 22)
		print("Saved " + str(shotNum)+yvalue+'.mp4 in animations')
	(te,dte,ne,dne,r, rprof,psinprof,times_ayc) = getClientData(shotNum)
	
	if yvalue == "all":
		createAnimation("te",te, shotNum)
		createAnimation("r",r, shotNum)
		createAnimation("ne",ne, shotNum)
		return
	elif yvalue == "te":
		createAnimation("te",te, shotNum)
	elif yvalue == "r":
		createAnimation("r",r, shotNum)
	elif yvalue == "ne":
		createAnimation("ne",ne, shotNum)
	else:
		return
def loadpkl(shotNum):
	#download pkl\
	if shotNum == "all":
		filename = 'MAST-U_pedestal.pkl'
	else:
		filename = 'output/MAST-U_pedestal_'+str(shotNum)+'.pkl'

	infile = open(filename, 'rb')
	global pkldata
	pkldata = pickle.load(infile)
	infile.close()
	return pkldata

def plotPedScal(shotNum):
	return


