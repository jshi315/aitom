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
from pyExcelerator import *
from xlrd import open_workbook
from xlutils.copy import copy

BETA = 0.5

class SSNR2D:

    '''
    input: images is a dictionary of 2d images indexed by keys
    '''
    def __init__(self, images, band_width_radius=1.0):

        im_f = {}
        for k in range(len(images)):
            im = images[k]
            im = fftshift(fftn(im))
            im_f[k] = im

        self.im_f = im_f        # fft transformed images
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

    def remove_from_set(self, k):
        assert k in self.im_f
        assert k in self.ks
        self.ks.remove(k)
        self.sum_v -= self.im_f[k]
        self.prod_sum_v -= self.im_f[k] * N.conj(self.im_f[k])

    def update_summary_statistics(self):
        sum_v = N.zeros(self.img_siz, dtype=N.complex)
        for k in self.ks:    sum_v += self.im_f[k]

        prod_sum_v = N.zeros(self.img_siz, dtype=N.complex)
        for k in self.ks:    prod_sum_v += self.im_f[k] * N.conj(self.im_f[k])

        self.sum_v = sum_v
        self.prod_sum_v = prod_sum_v

    def set_fft_mid_co(self):
        siz = self.img_siz
        assert(all(N.mod(siz, 1) == 0))
        assert(all(N.array(siz) > 0))

        mid_co = N.zeros(len(siz))

        # according to following code that uses numpy.fft.fftshift()
        for i in range(len(mid_co)):
            m = siz[i]
            mid_co[i] = N.floor(m/2)
        self.mid_co = mid_co

    def grid_displacement_to_center(self):
        size = N.array(self.img_siz, dtype=N.float)
        assert size.ndim == 1

        grid = N.mgrid[0:size[0], 0:size[1]]

        for dim in range(2):
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
        n = len(self.ks)
        avg = self.sum_v / n
        avg_abs_sq = N.real(  avg * N.conj( avg ) )
        var = (self.prod_sum_v - n * (avg * N.conj(avg))) / (n - 1)
        var = N.real(var)
        
        vol_rad = int( N.floor( N.min(self.img_siz) / 2.0 ) + 1)
        ssnr = N.zeros(vol_rad) + N.nan     # this is the SSNR of the AVERAGE image

        # the interpolation can also be performed using scipy.ndimage.interpolation.map_coordinates()
        for r in range(vol_rad):
            ind = self.rad_ind(r=r)         # in order to use it as an index or mask, must convert to a bool array, not integer array!!!!
            ind[N.logical_not(N.isfinite(avg))] = False
            ind[N.logical_not(N.isfinite(var))] = False

            if var[ind].sum() > 0:
                ssnr[r] = (n * avg_abs_sq[ind]).sum() / var[ind].sum()
            else:
                ssnr[r] = 0.0

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
    return images

def matching(ratio):
    imgs = {}
    result = []
    img_set = []
    true_image_set = []

    img_true = img("homo_imgs_1901.pickle")
    img_rotate = img("heter_imgs_1901.pickle")

    for i in range(500):
        img_set.append(img_snr10_rotate[i])
        if i < int(500*ratio):
            img_set.append(img_snr10_true[i])
            true_image_set.append(len(img_set)-1)

    #print true_image_set

    #img_siz = (40, 40)
    for i in range(len(img_set)):
        imgs[i] = img_set[i]

    s = SSNR2D(imgs)

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
            if e > 7.2:
                break
    center = center_num1
    result = []
    result.append(center)
    temp = []
    temp.append(center)
    
    while True:
        distance = []
        for i in range(len(img_set)):
            if i != center:
                temp.append(i)
                s.set_img_set(temp)
                distance.append(s.get_fsc_sum())
            else:
                distance.append(-1)
        center = distance.index(max(distance))
        temp = []
        temp.append(center)

        if center in result:
            break
        else:
            result.append(center)

    center = center_num2
    if center not in result:
        while True:
            distance = []
            for i in range(len(img_set)):
                if i != center:
                    temp.append(i)
                    s.set_img_set(temp)
                    distance.append(s.get_fsc_sum())
                else:
                    distance.append(-1)
            center = distance.index(max(distance))
            temp = []
            temp.append(center)

            if center in result:
                break
            else:
                result.append(center)

    end = time.clock()

    resolution = s.get_fsc_sum()
    t = end - start
    measure = cal_accuracy(result, true_image_set)
    print "Resolution = " + str(resolution)

    sheet = 2
    start_y = 1 + int(10*ratio-1)

    write_to_excel(sheet, 0, start_y, ratio)
    write_to_excel(sheet, 1, start_y, resolution)
    write_to_excel(sheet, 2, start_y, measure[0])
    write_to_excel(sheet, 3, start_y, measure[1])
    write_to_excel(sheet, 4, start_y, measure[2])
    write_to_excel(sheet, 5, start_y, t)

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
    matching(0.1)
    matching(0.2)
    matching(0.3)
    matching(0.4)
    matching(0.5)
    matching(0.6)
    matching(0.7)
    matching(0.8)
    matching(0.9)
    matching(1)
    
if __name__ == '__main__':
    main()

