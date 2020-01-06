#!/usr/bin/env python



'''
prepare true set of images using following steps:

given density map, generate projection image
then add noise

'''



import sys, json, pickle
import numpy as N

import tomominer.io.file as TIF
import tomominer.geometry.rotate as GR
import tomominer.image.optics.ctf as IOC

import util


def main():

    with open('true__op.json') as f:     op = json.load(f)          # read parameters
    with open(op['optics']) as f:      opt_op = json.load(f)
    ctf_op = opt_op['ctf']


    v = TIF.pickle_load(op['maps file'])[op['pid']]              # load a 3D density map
    if not op['intensity_positive']:    v = -v

    v = GR.rotate_pad_zero(v, angle=op['rotate_angle'], loc_r=op['translation'])       # rotate image

    p = N.squeeze(v.sum(axis=2))            # make a projection image along z-axis

    ctf = IOC.create(size=p.shape, Dz=ctf_op['Dz'], pix_size=ctf_op['pix_size'], voltage=ctf_op['voltage'], Cs=ctf_op['Cs'], sigma=ctf_op['sigma'])['ctf']

    p_var = p.var()
    n_var = p_var / opt_op['snr']


    # simulate a number of images
    imgs = []
    for i in range(op['image_num']):
        print '\r', i, '                   ',               ;           sys.stdout.flush()
        imgs.append(util.simulate(p=p, ctf=ctf, noise_total_var=n_var))


    with open(op['images_out'], 'wb') as f:     pickle.dump(imgs, f, protocol=-1)



if __name__ == '__main__':
    main()




'''
# to export images in png format, use following script

if False:
    fn = 'true__out__images.pickle'
    out_dir = './true__out__images'
else:
    fn = 'rotation_var__out__images.pickle'
    out_dir = './rotation_var__out__images'

import shutil
shutil.rmtree(out_dir)

import os
os.makedirs(out_dir)

import pickle
with open(fn, 'rb') as f:      imgs = pickle.load(f) 

import png
import numpy as N
for i, g in enumerate(imgs):
    g = g - g.min()
    g /= g.max()
    g *= 65535
    g = g.astype(N.uint16)
    png.from_array(g, 'L').save(os.path.join(out_dir, '%03d.png'%(i,)))



'''



'''
~/ln/frequent_structure/data/out/method/resolution-guided-averaging/2d/simulation/scripts/data_prepare/true.py
'''






