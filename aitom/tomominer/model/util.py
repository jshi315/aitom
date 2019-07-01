

'''
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
'''



import numpy as N
import aitom.tomominer.image.vol.util as gv
import aitom.tomominer.geometry.rotate as GR

def boundary_mask(shape):
    shape = N.array(shape)
    m = N.zeros(shape)
    m[0, :, :] = 1
    m[(-1), :, :] = 1
    m[:, 0, :] = 1
    m[:, (-1), :] = 1
    m[:, :, 0] = 1
    m[:, :, (-1)] = 1
    return m