from matplotlib import pyplot as plt
from distgen import Generator
import yaml
import sys
import os
import h5py
import numpy as np
from PIL import Image

dirname = os.path.dirname(__file__)
print(dirname)
input_file = dirname + '/data/rad.gaussian.in.yaml'
print(input_file)
gen = Generator(input_file, verbose=0) 
gen.run()
pg = gen.particles

pg.plot('x', 'px')
fig1=plt.gcf()
fig1.savefig('test.png')

gen = Generator(dirname + '/data/jpeg.brule.image.in.yaml',verbose=0)
gen.run()
pg = gen.particles
pltx = pg.x
plty = pg.y
print(pltx)
print(plty)
fig,ax = plt.subplots()
#plt.rcParams['agg.path.chunksize']=100000
figg=plt.hist2d(pltx,plty)
figg=plt.gcf()
#ax.plot(pltx,plty)
#ax.set_axis_off()
#fig.savefig('test.jpeg', bbox_inches='tight')
#plt.rcParams['agg.path.chunksize']=10000
#extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

#fig.savefig('testent.png', bbox_inches=extent)
#fig.savefig('testight.png',bbox_inches='tight', pad_inches=0.0)
#fig1=plt.gcf()
#figg.show()
figg.savefig('testold.png', bbox_inches='tight')