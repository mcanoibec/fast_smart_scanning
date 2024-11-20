#This script separates into convolution images with dimensions kernel_size x kernel_size
import numpy as np
from tqdm import tqdm
def conv_imgs(matrix,kernel_size, idx=[]):

    if len(idx)<=0:
        ix=0
        iy=0
        samps=np.zeros((1, kernel_size,kernel_size))

        for j in tqdm(np.arange(len(matrix)-kernel_size+1)):
            iy=j
            for i in np.arange(len(matrix[0])-kernel_size+1):
                ix=i
                samp=matrix[iy:iy+kernel_size,ix:ix+kernel_size]
                samp=np.expand_dims(samp, axis=0)
                samps=np.concatenate((samps,samp), axis=0)
        samps=samps[1:,:,:]
    else:
        if kernel_size%2 == 0:
            extra=0
        else:
            extra=1
        half_kernel=int(np.round((kernel_size-1)/2))

        ix=0
        iy=0
        samps=np.zeros((1, kernel_size,kernel_size))

        for j in tqdm(np.arange(len(idx[0,:]))):
            iy=idx[0,j]
            ix=idx[1,j]
            samp=matrix[iy-half_kernel:iy+half_kernel+extra,ix-half_kernel:ix+half_kernel+extra]
            samp=np.expand_dims(samp, axis=0)
            samps=np.concatenate((samps,samp), axis=0)
        samps=samps[1:,:,:]
    return samps