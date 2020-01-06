


'''
utility functions

~/ln/frequent_structure/data/out/method/resolution-guided-averaging/2d/simulation/scripts/data_prepare/util.py
'''


import numpy as N

from numpy.fft import fftn, ifftn, fftshift, ifftshift



'''
generate simulated projection image, according to

line 843 of       ~/ln/frequent_structure/code/GenerateSimulationMap.m
'''

def simulate(p, ctf, noise_total_var, proj_var_weight=0.5):
    p = N.copy(p)           # need to make a copy so that the original is unchanged
    
    proj_var = noise_total_var * proj_var_weight

    p += N.random.normal(scale=N.sqrt(proj_var), size=p.shape)         # adding noise

    pf = fftshift(fftn(p))
    p = N.real(ifftn(ifftshift(pf * ctf)))          # convolute with CTF,    todo, need to also apply a sphere mask

    mtf_var_weight = 1 - proj_var_weight
    mtf_var = noise_total_var * mtf_var_weight

    p += N.random.normal(scale=N.sqrt(mtf_var), size=p.shape)         # adding noise

    # todo: add bandpass filter

    return p


