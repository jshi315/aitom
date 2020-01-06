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
from random import choice
from pyExcelerator import *
from xlrd import open_workbook
from xlutils.copy import copy

ITERATION = 20
NUM_FIRST_G = 40 # after testing
BETA = 0.5

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

def find_img_index(vector):
    find = 1
    index = [i for i,j in enumerate(vector) if j == find]
    return index

def genetic_algorithm():
    imgs = {}
    masks = {}
    #img_siz = (40, 40)
    img_set = []
    mask = []
    t = [] # time
    measure = []
    true_image_set = []
    resolution = [] #record the resolution after every iteration

    img_set = img("ribosome_annealing.pickle")[0]
    mask = img("ribosome_annealing.pickle")[1]
    judge = img("ribosome_annealing.pickle")[2]
    true_image_set = img("ribosome_annealing.pickle")[3]

    dimension = len(img_set)
    maxi = dimension // 2

    for i in range(dimension):
        imgs[i] = img_set[i]
        masks[i] = mask[i]
    s = SSNR3D(imgs, masks)

    start = time.clock()

    img_num = range(dimension)
    g_zero = [0] * dimension

    initial = []
    generation = []
    generation_1 = [] #nest generation
    result_list = []

    # At first, 10 candidate solutions. Each solution has 200 dimensions.
    for i in range(NUM_FIRST_G):
        initial.append(random.sample(img_num, maxi))
    for i in range(NUM_FIRST_G):
        for j in range(maxi):
            g_zero[initial[i][j]] = 1
        generation.append(g_zero)
        g_zero= []
        g_zero = [0] * dimension

    iteration_list = []
    temp = []

    for i in range(NUM_FIRST_G):
        s.set_img_set(find_img_index(generation[i]))
        temp.append(s.get_fsc_sum())

    iteration_list.append(max(temp))
    result = find_img_index(generation[temp.index(max(temp))])
    result_list.append(result)

    end = time.clock()
    
    # first iteration
    resolution.append(max(temp))
    t.append(end-start)
    measure.append(cal_accuracy(result, true_image_set))
    print "resolution = " + str(max(temp))
 
    for i in range(ITERATION-1): #max number of iterations
        #There are always a half of this generation chosen to be alive and a half mutated
        temp = []
        #natural selection
        for j in range(len(generation)):
            s.set_img_set(find_img_index(generation[j]))
            temp.append(s.get_fsc_sum())

        temp_best = []
        for j in range(len(generation)//2):
            temp_best.append(generation[temp.index(max(temp))])
            temp[temp.index(max(temp))] = 0

        #mutation process(crossover operation)
        P_new = []
        while len(P_new) < len(temp_best):
            P = random.sample(temp_best, 2)
            pl1 = P[0][:dimension//2]
            pr1 = P[0][dimension//2:]
            pl2 = P[1][:dimension//2]
            pr2 = P[1][dimension//2:]
            P_new1 = pl1 + pr2
            P_new2 = pl2 + pr1
            if random.random() < 0.5:
                ran = random.randint(0, dimension-1)
                if P_new1[ran] == 0:
                    P_new1[ran] = 1
                else:
                    P_new1[ran] = 0
                ran = random.randint(0, dimension-1)
                if P_new2[ran] == 0:
                    P_new2[ran] = 1
                else:
                    P_new2[ran] = 0

            P_new.append(P_new1)
            P_new.append(P_new2)
          
        generation_1 = P_new + generation
        
        temp = []
        for j in generation_1:
            if len(find_img_index(j)) > maxi:
                temp.append(j)
        for j in temp:
            generation_1.remove(j)

        best = 0
        for j in generation_1:
            s.set_img_set(find_img_index(j))
            best_new = s.get_fsc_sum()
            if best_new > best:
                best = best_new
                result = find_img_index(j)
        #print best
        result_list.append(result)
        iteration_list.append(best)

        end = time.clock()

        resolution.append(best)
        t.append(end-start)
        measure.append(cal_accuracy(result, true_image_set))
        print "resolution = " + str(best)

        generation = generation_1
        generation_1 = []

    start_y = 1
    sheet = 1

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
    genetic_algorithm()

if __name__ == '__main__':
    main()

