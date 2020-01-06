#!/usr/bin/env python



'''
similiar to true.py but
prepare set of images of randomly rotated objects
'''

import sys, json, pickle
import numpy as N

import tomominer.io.file as TIF
import tomominer.geometry.rotate as GR
import tomominer.image.optics.ctf as IOC

import util


def main():

    with open('rotation_var__op.json') as f:     op = json.load(f)          # read parameters
    with open(op['true']) as f:     opt = json.load(f)

    with open(opt['optics']) as f:      opt_op = json.load(f)
    ctf_op = opt_op['ctf']


    v = TIF.pickle_load(opt['maps file'])[opt['pid']]              # load a 3D density map
    if not opt['intensity_positive']:    v = -v

    p = N.squeeze(v.sum(axis=2))            # make a projection image along z-axis

    ctf = IOC.create(size=p.shape, Dz=ctf_op['Dz'], pix_size=ctf_op['pix_size'], voltage=ctf_op['voltage'], Cs=ctf_op['Cs'], sigma=ctf_op['sigma'])['ctf']

    p_var = p.var()
    n_var = p_var / opt_op['snr']


    # simulate a number of images
    imgs = []
    for i in range(op['image_num']):
        print '\r', i, '                   ',               ;           sys.stdout.flush()
        ang_t = N.array(opt['rotate_angle']) + N.random.normal(size=3) * op['rotation_angle_standard_deviation']
        loc_t = (N.array(opt['translation']) + N.random.normal(size=3) * op['translation_standard_deviation']) if ('translation_standard_deviation' in opt) else N.zeros(3)
        vr = GR.rotate_pad_zero(v, angle=ang_t, loc_r=loc_t)       # rotate image
        pr = N.squeeze(vr.sum(axis=2))
       
        imgs.append(util.simulate(p=pr, ctf=ctf, noise_total_var=n_var))


    with open(op['images_out'], 'wb') as f:     pickle.dump(imgs, f, protocol=-2)


    
    


if __name__ == '__main__':
    main()




'''
# to export images in png format, use following script


'''






'''
~/ln/frequent_structure/data/out/method/resolution-guided-averaging/2d/simulation/scripts/data_prepare/rotation_var.py
'''






