#! /usr/bin/env python

import sys
import pylab as pl
import numpy as np

if sys.argv.__len__()<2:
	print 'Please provide the name of a row-major matrix to display'
	sys.exit()

filename = sys.argv[1]
f=open(filename,'r')
strfile=f.read()

mat=np.array(strfile.split(),dtype=np.float)	#interpret split string as floats
N=mat[0]										#first number is the simension
mat=mat[1:] 									#clip it off before reshaping
mat=mat.reshape((N,N),order='C')  				#C is row-major

pl.imshow(mat)
pl.title(filename)
pl.show()
