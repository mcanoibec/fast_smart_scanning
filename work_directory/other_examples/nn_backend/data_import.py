#Data Import Functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import sys

sys.path.append("D:\personal_scripts")
from import_xyz_img import import_xyz_img
from conv_imgs import conv_imgs
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import cm, colormaps
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, concatenate
from keras.optimizers import Adam
from keras.utils import set_random_seed
from scipy.stats import norm
import seaborn as sns
import math
import random
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from scipy import ndimage
from nn_backend.modified_loss_functions import cube_msle, linear_msle
from skimage import morphology

def import_curves_from_folder(curves_file, curves_max_limit=1000):
    m_vec=[]
    zetas=np.arange(curves_max_limit)
    curves_nu=np.zeros((1,curves_max_limit))
    num=0
    last=0

    onlyfiles = [f for f in listdir(curves_file) if isfile(join(curves_file, f))]

    for i in tqdm(onlyfiles, 
                    desc="Loading curves…", 
                    ascii=False, ncols=75):
        file = np.loadtxt(rf'{curves_file}\{i}', skiprows=72)

        file[:,0] = file[:,0]*1e9

        file=np.flip(file,axis=0)

        for n in np.arange(len(file))+1:
            if file[-n,0]<file[-n-1,0]:
                file=file[-n:,:]
                break

        m=(file[10,1]-file[0,1])/(file[10,0]-file[0,0])
        m_vec=np.append(m_vec,m)
        b=file[0,1]-(m*file[0,0])
        slope=m*file[:,0]+b

        baseline=np.mean(file[np.round(0.7*len(file)).astype(int):,1])
        base_std=np.std(file[np.round(0.7*len(file)).astype(int):,1])
        base_thresh=baseline-2*base_std

        for k in np.arange(0,len(file)):        
            if base_thresh>slope[(k)]:
                ZContact=file[(k),0]
                break


        file[:,0]=file[:,0]-ZContact


        maxx=np.argmax(file[:,2])
        file=file[maxx:,:]

        file[:,0]=np.round(file[:,0])
        file=pd.DataFrame(file)
        file.drop_duplicates(subset=0,inplace=True)
        file=np.array(file)
        ccc=file[:,2]
        zzz=file[:,0]
        projection = np.full((len(zetas)), np.nan)
        indices = np.where(np.isin(zetas, zzz.astype(int)))[0]
        projection[indices] = ccc[np.isin(np.round(zzz), zetas)]
        projection=np.expand_dims(projection, axis=0)
        curves_nu=np.concatenate((curves_nu, projection), axis=0)
    curves_nu=curves_nu[1:,:]
    curves_nu=pd.DataFrame(curves_nu)
    curves_nu=curves_nu.interpolate(axis=1)
    curves_nu=curves_nu.bfill(axis=1)

    return curves_nu, m_vec

def curve_calibration(curves, m_vec, K, G, uac):
    mean_m=abs(np.mean(m_vec))
    A=(2** 0.5)*uac
    curves=((4*K*(2**0.5))/(A**2))*((curves)/(mean_m*G))
    return curves

def matrix_cut(matrix, end_dimension):
    #Cuts a square 
    if len(matrix)<end_dimension:
        print('Error: Exit matrix size is larger than the original dimensions')
        return matrix
    else:
        side=int(len(matrix)/2)
        dim_2=int(end_dimension/2)
        matrix=matrix[side-dim_2:side+dim_2,side-dim_2:side+dim_2]
        return matrix
    
def data_augment(cv_mat):
    Cv_rot90=np.rot90(cv_mat,k=1,axes=(1,2))
    Cv_rot180=np.rot90(cv_mat,k=2,axes=(1,2))
    Cv_rot270=np.rot90(cv_mat,k=3,axes=(1,2))

    Cv_ud=np.flip(cv_mat,axis=1)
    Cv_lr=np.flip(cv_mat,axis=2)
    Cv_aug=np.concatenate((cv_mat, Cv_rot90,Cv_rot180,Cv_rot270,Cv_ud, Cv_lr), axis=0)
    return Cv_aug

def mask_from_topo(topo,binary_threshold,object_size,hole_size):
    bin=np.array(topo)
    bin_thresh=np.max(topo)*binary_threshold
    bin[bin<=bin_thresh]=0
    bin[bin>0]=1
    bin=bin.astype(bool)
    bin=bin.reshape(int(math.sqrt(len(topo))),int(math.sqrt(len(topo))))
    bin = morphology.remove_small_objects(bin, object_size)
    bin = morphology.remove_small_holes(bin, hole_size)
    return bin