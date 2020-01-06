'''
a class for calculating ssnr and fsc of average of a set of 2D images
'''
from __future__ import division
import copy as C
import numpy as N
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import pickle
import time
import random
import math
from pyExcelerator import *
from xlrd import open_workbook
from xlutils.copy import copy

BETA = 0.5
ITERATION = 40

class SSNR3D:

    '''
    input: images is a dictionary of 3D images indexed by keys; masks is a dictionary of 3D images of binary values
    '''
    def __init__(self, images, masks, band_width_radius=1.0):

        im_f = {}
        for k in images:
            im = images[k]
            im = fftshift(fftn(im))
            im_f[k] = im

        self.im_f = im_f        # fft transformed images
        self.ms = masks         # masks
        self.ks = set()
        self.img_siz = im_f[k].shape
        self.set_fft_mid_co()
        self.set_rad()
        self.band_width_radius = band_width_radius
        
    def set_img_set(self, ks):
        for k in ks:    assert k in self.im_f
        self.ks = C.deepcopy(ks)
        self.ks = set(self.ks)
        self.update_summary_statistics()

    def add_to_set(self, k):
        assert k in self.im_f
        assert k not in self.ks
        self.ks.add(k)
        self.sum_v += self.im_f[k]
        self.prod_sum_v += self.im_f[k] * N.conj(self.im_f[k])
        self.mask_sum_v += self.ms[k]

    def remove_from_set(self, k):
        assert k in self.im_f
        assert k in self.ks
        self.ks.remove(k)
        self.sum_v -= self.im_f[k]
        self.prod_sum_v -= self.im_f[k] * N.conj(self.im_f[k])
        self.mask_sum_v -= self.ms[k]

    def update_summary_statistics(self):
        sum_v = N.zeros(self.img_siz, dtype=N.complex)
        for k in self.ks:    sum_v += self.im_f[k]

        prod_sum_v = N.zeros(self.img_siz, dtype=N.complex)
        for k in self.ks:    prod_sum_v += self.im_f[k] * N.conj(self.im_f[k])

        mask_sum_v = N.zeros(self.img_siz, dtype=float)
        for k in self.ks:    mask_sum_v += self.ms[k]

        self.sum_v = sum_v
        self.prod_sum_v = prod_sum_v
        self.mask_sum_v = mask_sum_v

    def set_fft_mid_co(self):
        siz = self.img_siz
        assert(N.all(N.mod(siz, 1) == 0))
        assert(N.all(N.array(siz) > 0))

        mid_co = N.zeros(len(siz))

        # according to following code that uses numpy.fft.fftshift()
        for i in range(len(mid_co)):
            m = siz[i]
            mid_co[i] = N.floor(m/2)
        self.mid_co = mid_co

    def grid_displacement_to_center(self):
        size = N.array(self.img_siz, dtype=N.float)
        assert size.ndim == 1

        grid = N.mgrid[0:size[0], 0:size[1], 0:size[2]]

        for dim in range(3):
            grid[dim, :, :] -= self.mid_co[dim]

        return grid

    def grid_distance_sq_to_center(self, grid):
        dist_sq = N.zeros(grid.shape[1:])
        if grid.ndim == 4:
            for dim in range(3):
                dist_sq += N.squeeze(grid[dim, :, :, :]) ** 2
        elif grid.ndim == 3:
            for dim in range(2):
                dist_sq += N.squeeze(grid[dim, :, :]) ** 2
        else:
            assert False

        return dist_sq

    def grid_distance_to_center(self, grid):
        dist_sq = self.grid_distance_sq_to_center(grid)
        return N.sqrt(dist_sq)

    # get a volume containing radius
    def set_rad(self):
        grid = self.grid_displacement_to_center()
        self.rad = self.grid_distance_to_center(grid)

    # get index within certain frequency band
    def rad_ind(self, r):
        return ( abs(self.rad - r) <= self.band_width_radius )

    def get_ssnr(self):
        ind = self.mask_sum_v > 2
        avg = N.zeros(self.sum_v.shape, dtype=N.complex) + N.nan
        avg[ind] = self.sum_v[ind] / self.mask_sum_v[ind]

        avg_abs_sq = N.zeros(self.sum_v.shape, dtype=N.complex) + N.nan
        avg_abs_sq[ind] = avg[ind] * N.conj( avg[ind] )

        var = N.zeros(self.sum_v.shape, dtype=N.complex) + N.nan
        var[ind] = (self.prod_sum_v[ind] - self.mask_sum_v[ind] * avg_abs_sq[ind]) / (self.mask_sum_v[ind] - 1)
        var = N.real(var)
        
        vol_rad = int( N.floor( N.min(self.img_siz) / 2.0 ) + 1)
        ssnr = N.zeros(vol_rad) + N.nan     # this is the SSNR of the AVERAGE image

        # the interpolation can also be performed using scipy.ndimage.interpolation.map_coordinates()
        for r in range(vol_rad):
            ind = self.rad_ind(r=r)         # in order to use it as an index or mask, must convert to a bool array, not integer array!!!!
            ind[N.logical_not(N.isfinite(avg))] = False
            ind[N.logical_not(N.isfinite(var))] = False

            if var[ind].sum() > 0:
                ssnr_t = (self.mask_sum_v[ind] * avg_abs_sq[ind]).sum() / var[ind].sum()
            else:
                ssnr_t = 0.0
            ssnr[r] = N.real(ssnr_t)

        assert N.all(N.isfinite(ssnr))
        return ssnr

    def get_fsc(self):
        ssnr = self.get_ssnr()
        fsc = ssnr / (2.0 + ssnr)
        return fsc

    '''
    this is the objective function to be minimized
    '''
    def get_fsc_sum(self):
        return self.get_fsc().sum()


def img(fileName):
    pkl_file = open(fileName, 'rb')
    images = {}
    images = pickle.load(pkl_file)
    pkl_file.close()
    img_set = []
    mask = []
    judge = [] # true or rotate
    true_image_set = []
    data = []

    for n in images:
        img_set.append(n['v'])
        mask.append(n['m'])
        judge.append(n['i'])

    for n in range(len(judge)):
        if judge[n] == True:
            true_image_set.append(n)    

    data.append(img_set)
    data.append(mask)
    data.append(judge)
    data.append(true_image_set)

    return data

# don't know the number of target images
def simulated_annealing():
    imgs = {}
    masks = {}

    t = [] # time
    measure = []
    true_image_set = []
    resolution = [] #record the resolution after every iteration

    img_set = img("ribosome_annealing.pickle")[0]
    mask = img("ribosome_annealing.pickle")[1]
    judge = img("ribosome_annealing.pickle")[2]
    true_image_set = img("ribosome_annealing.pickle")[3]
            
    #img_siz = (40, 40)
    for i in range(len(img_set)):
        imgs[i] = img_set[i]
        masks[i] = mask[i]

    s = SSNR3D(imgs, masks)

    start = time.clock()

    while True:
        result = []
        center_num1 = random.randrange(0, len(img_set), 1)
        center_num2 = random.randrange(0, len(img_set), 1)
        if center_num1 != center_num2:
            result.append(center_num1)
            result.append(center_num2)
            s.set_img_set(result)
            e = s.get_fsc_sum()
            if e > 6.8:
                break
    print result

    s0 = center_num2
    sn = s0
    k = 0
    kmax = len(img_set)

    # first iteration

    while (k < kmax) and (len(result) < kmax//2):
        while True:
            sn = sn + 1
            if sn == kmax:
                sn = 0
            if sn not in result:
                break

        s.add_to_set(sn)
        en = s.get_fsc_sum()
        s.remove_from_set(sn)

        if en > e:
            result.append(sn)
            s.set_img_set(result)
            e = en
        elif math.exp(-abs(en-e)/(k+1)) < random.random(): # after testing, choose 350 (need >50)
            result.append(sn)
            s.set_img_set(result)
            #e = en
        k+=1
    s_record = sn #record "sn" and when next iteration, start from sn

    for sn in result:
        s.remove_from_set(sn)
        en = s.get_fsc_sum()
        s.add_to_set(sn)
        if en > e:
            result.remove(sn)
            s.set_img_set(result)
            e = en

    resolution.append(e)
    measure.append(cal_accuracy(result, true_image_set))

    end = time.clock()
    t.append(end-start)
    print "Resolution = " + str(e)
    
    # iterations

    for i in range(ITERATION-1):
        k = 0
        sn = s_record
        while (k < kmax) and (len(result) < kmax//2):
            while True:
                sn += 1
                if sn == kmax:
                    sn = 0
                if sn not in result:
                    break

            s.add_to_set(sn)
            en = s.get_fsc_sum()
            s.remove_from_set(sn)
            if en > e: 
                result.append(sn)
                s.set_img_set(result)
                e = en
            elif math.exp(-abs(en-e)/(k+1)) < random.random():
                result.append(sn)
                s.set_img_set(result)
                #e = en
            k+=1
        s_record = sn

        for sn in result:
            s.remove_from_set(sn)
            en = s.get_fsc_sum()
            s.add_to_set(sn)
            if en > e:
                result.remove(sn)
                s.set_img_set(result)
                e = en
        resolution.append(e)
        measure.append(cal_accuracy(result, true_image_set))

        end = time.clock()
        t.append(end-start)

    print "Resolution = " + str(e)
    print "Time = " + str(end-start)

    start_y = 1
    sheet = 0

    for i in range(start_y, ITERATION+start_y):
        write_to_excel(sheet, 1, i, resolution[i-start_y])
        write_to_excel(sheet, 2, i, measure[i-start_y][0])
        write_to_excel(sheet, 3, i, measure[i-start_y][1])
        write_to_excel(sheet, 4, i, measure[i-start_y][2])
        write_to_excel(sheet, 5, i, t[i-start_y])
        if i == start_y:
            write_to_excel(sheet, 0, i, len(true_image_set)/(len(img_set)-len(true_image_set)))
            write_to_excel(sheet, 6, i, len(img_set))

def write_to_excel(sheet, x, y, value): #x->col, y->row
    rb = open_workbook('record.xls')
    rs = rb.sheet_by_index(sheet)
    wb = copy(rb)
    ws = wb.get_sheet(sheet)
    ws.write(y, x, value)
    wb.save('record.xls')

def cal_accuracy(result, true_image_set):
    return_list = []
    n_right = 0
    n_wrong = 0
    for i in result:
        if i in true_image_set:
            n_right += 1
        else:
            n_wrong += 1
    precision = n_right/len(result)
    recall = n_right/len(true_image_set)
    print "precision = " + str(precision*100) + " %"
    print "recall = " + str(recall*100) + " %"
    if (precision != 0.0) and (recall != 0.0):
        f = (1+pow(BETA,2)) * (precision*recall) / (precision*pow(BETA,2)+recall)
    else:
        f = 0.0
    return_list.append(precision)
    return_list.append(recall)
    return_list.append(f)

    return return_list

def main():
    simulated_annealing()
    
if __name__ == '__main__':
    main()


