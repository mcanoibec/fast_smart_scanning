# %% [markdown]
# ## Neural network playing ground (Small Methods Imgs)
# ### Nov 2023 - Code by Mauricio Cano Galván

# %% [markdown]
# Library Import

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import copy


from nn_backend.import_xyz_img import import_xyz_img
from nn_backend.conv_imgs import conv_imgs
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
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from scipy import ndimage
from nn_backend.modified_loss_functions import cube_msle, linear_msle
from skimage import morphology
from nn_backend.data_import import import_curves_from_folder, curve_calibration, matrix_cut, data_augment, mask_from_topo
from nn_backend.nn_for_ep import nn_for_ep
from nn_backend.post_process import prediction_reconstruction_with_train, img_export, histogram_export
from skimage.metrics import structural_similarity as ssim


viridis = colormaps['viridis']
newcolors = viridis(np.linspace(0, 1, 256))
black = np.array([0, 0, 0, 1])
newcolors[0:33, :] = black
newcmp = ListedColormap(newcolors)
del black, newcolors, viridis

# %%
#Constant Setup
pctg=3.49


cutout_size=9

#Graphics (1 to show images, 0 to not)
kernel_visualization=0 
input_imgs_view=1
train_curve=0

#NN Setup
kernel_dim=3
iterations=300
batch_sz=200

#Other parameters
#topo_area_dim=9
use_last_curves=1

name='efm8'
curves_file=rf'c:\Users\mcano\Code\rawdata\preprocessed curves\efm8_curves_calibrated.npy'
topography_file=rf'c:\Users\mcano\Code\rawdata\img_data\EFM8\Topography Flattened EFM8.txt'
ref_file=rf"c:\Users\mcano\Code\rawdata\img_data\EFM8\Map_Nanofilament.txt"
raw_curves_folder=0
curves_max_limit=1000
calibrate_curves=1
nu_dim=120
ref_flip=1
topo_flip=0

# %% [markdown]
# Data Import

# %%
#Curve import
if not raw_curves_folder:
    curves_projected=pd.DataFrame(np.load(rf'{curves_file}'))
    curves_projected=curves_projected.interpolate(axis=1)
    curves_projected=curves_projected.bfill(axis=1)
else:
    curves_projected, m_vec=import_curves_from_folder(curves_file)

    if calibrate_curves:
        K=0.718
        G=200
        uac=3
        curves_projected=curve_calibration(curves_projected, m_vec, K, G, uac)



# %%
#curves_name=rf'efm9_curves_calibrated'
#np.save('{curves_name}.npy', curves_projected)

# %%
#------Import the label data (εp map), as a matrix first and then transformed into a DataFrame.------#
#----------The data is then cleaned to remove outliers
y_mat, ax_ep, ay_ep= import_xyz_img(ref_file)
if ref_flip:
    y_mat=np.flipud(y_mat)
y_mat=matrix_cut(y_mat,nu_dim)
y=pd.DataFrame(np.reshape(y_mat,nu_dim**2))
#------Manual noise removal in label *ONLY FOR EFM9*------#

#Logarithmic sampling
maxlen=1713
curve_height_setpoint=20
curve_nsamples=12


zs=np.round((np.logspace(start=0,stop=math.log10(maxlen),num=curve_nsamples,endpoint=False)+curve_height_setpoint)).astype(int)
samples = []
[samples.append(x) for x in zs if x not in samples]


#New curves
curves=(np.array(curves_projected)).reshape(128,128,len(curves_projected.iloc[0,:]))
curves=matrix_cut(curves, nu_dim)
curves_sampled=curves[:,:,samples]

#Topography import
T_mat, x, y2 = import_xyz_img(topography_file)
if topo_flip:
    T_mat=np.flipud(T_mat)
T_mat=matrix_cut(T_mat, nu_dim)
T_mat=np.expand_dims(T_mat, axis=2)
T=T_mat.reshape(nu_dim**2)
#Put together training dataset
X_mat=np.concatenate((T_mat, curves_sampled), axis=2)

aa=["Z = "+str(x)+" nm" for x in samples]
inputs=["Topography"]
inputs=np.append(inputs, aa)

X=np.reshape(X_mat,(nu_dim**2,len(X_mat[0,0,:])))
X=pd.DataFrame(X, columns=inputs)




# %% [markdown]
# Data Pre-Processing

# %%
#EFM9: binary_threshold=0.01, object_size=3, hole_size=30, m_thresh=0.30
#EFM8: binary_threshold=0.41, object_size=3, hole_size=30, m_thresh=0.77
#EFM7: binary_threshold=0.095, object_size=4, hole_size=15, m_thresh=0.44
#EFVM: binary_threshold=0.05, object_size=2, hole_size=2, m_thresh=0.785

binary_threshold=0.41
object_size=3
hole_size=30
m_ep_thresh=0.77


bin=mask_from_topo(T,binary_threshold,object_size,hole_size)

cell=bin.reshape(nu_dim**2,1)
roi_index=pd.DataFrame(copy(cell))
roi_index=roi_index.index[(roi_index[0])]


mask_ep=pd.DataFrame(copy(y))
th=float(mask_ep.quantile(m_ep_thresh))

mask_ep[(mask_ep[0]<=th)]=0
mask_ep[(mask_ep[0]!=0)]=1
mask_ep=cell*mask_ep
mask_ep_im=np.reshape(np.array(mask_ep), ((nu_dim),(nu_dim)))

cell_index=mask_ep.index[mask_ep[0]>0].tolist()


fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(1,2,1)
ax.imshow(bin, cmap='gray')
ax.set_title('ROI mask')
ax.axis('off')
ax=fig.add_subplot(1,2,2)
ax.imshow(mask_ep_im)
ax.axis('off')
ax.set_title('Final Train Mask')


# %%
#------Cell separation------#
y_cell=pd.DataFrame(y.iloc[cell_index])
X_cell=pd.DataFrame(X.iloc[cell_index])
y_roi=pd.DataFrame(y.iloc[roi_index])
X_roi=pd.DataFrame(X.iloc[roi_index])


#------Data Normalization------#
curve_scale=StandardScaler()
curve_scale.fit(X_cell)
Xn_cell=pd.DataFrame(curve_scale.transform(X_cell), columns=inputs)
Xn_roi=pd.DataFrame(curve_scale.transform(X_roi), columns=inputs)

# %%

#------Data Normalization------#
curve_scale=StandardScaler()
curve_scale.fit(X_cell)
Xn_cell=pd.DataFrame(curve_scale.transform(X_cell), columns=inputs)

# %%

topo_mat, x, y2 = import_xyz_img(topography_file)
if topo_flip:
    topo_mat=np.flipud(topo_mat)
topo=topo_mat.reshape(len(topo_mat)**2,1)
topo_scale=StandardScaler()
topo_scale.fit(topo)
topo_n=topo_scale.transform(topo)
topo_n_mat=topo_n.reshape(len(topo_mat),len(topo_mat),1)

Cv_mat=conv_imgs(topo_n_mat[:,:,0], cutout_size)
Cv_mat=Cv_mat.reshape(int(math.sqrt(len(Cv_mat))),int(math.sqrt(len(Cv_mat))),cutout_size,cutout_size)
Cv_mat=matrix_cut(Cv_mat, nu_dim)
Cv_mat=Cv_mat.reshape(nu_dim*nu_dim,cutout_size,cutout_size)
Cv_mat=np.expand_dims(Cv_mat, axis=-1)



Cv_cell_mat=Cv_mat[cell_index,:,:,:]
Cv_roi_mat=Cv_mat[roi_index,:,:,:]


# %% [markdown]
# Test-Train Separation

# %%
pctg_list=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
rsqrs=[]
maes=[]
ssims=[]
for ppp in tqdm(pctg_list, desc="Progress final"):
    sss = ShuffleSplit(n_splits=1, test_size=1-(ppp/100), random_state=1)
    sss.get_n_splits(Xn_cell, y_cell)
    train_index, test_index = next(sss.split(Xn_cell, y_cell))
    train_index=np.sort(train_index)  
    test_index=np.sort(test_index)  



    X_train, X_test = Xn_cell.iloc[train_index], Xn_cell.iloc[test_index] 
    y_train, y_test = y_cell.iloc[train_index], y_cell.iloc[test_index]

    Cv_train, Cv_test = Cv_cell_mat[train_index,:,:], Cv_cell_mat[test_index,:,:] 
    Cv_train_aug=data_augment(Cv_train)


    N_aug=int(len(Cv_train_aug[:,0,0,0])/len(Cv_train[:,0,0,0]))

    X_train_aug=copy(X_train)
    y_train_aug=copy(y_train)
    for i in np.arange(N_aug-1):
        X_train_aug=pd.DataFrame(np.concatenate((X_train_aug, X_train), axis=0))
        y_train_aug=pd.DataFrame(np.concatenate((y_train_aug, y_train), axis=0))

    # %% [markdown]
    # Network Architecture

    # %%
    #------Example Keras Network------#
    reg, histo=nn_for_ep(X_train_aug,Cv_train_aug,y_train_aug,X_test,Cv_test,y_test, verb=0)

    # %% [markdown]
    # Results

    # %%
    res=reg.predict([Xn_roi, Cv_roi_mat])

    # %% [markdown]
    # Results visualization

    # %%
    full_vec_roi=np.zeros(nu_dim**2)+1
    for i in np.arange(len(roi_index)):
        full_vec_roi[roi_index[i]]=res[i]



    for i in y.index[np.array(cell_index)[train_index]]:
        full_vec_roi[i]=y.iloc[i]

    full_img_roi=np.reshape(full_vec_roi,(nu_dim,nu_dim))
    full_vec_roi=pd.DataFrame(full_vec_roi)
    full_roi=copy(full_vec_roi)
    full_cell=pd.DataFrame(full_roi.iloc[roi_index])

    #Metrics calculations
    res_cell=np.array(full_vec_roi.iloc[cell_index])
    full_vec_cell=np.zeros(nu_dim**2)+1
    for i in np.arange(len(cell_index)):
        full_vec_cell[cell_index[i]]=res_cell[i]
    full_img_cell=np.reshape(full_vec_cell,(nu_dim,nu_dim))

    error_map=np.subtract(full_img_cell,y_mat)
    error_map_r=(abs(np.subtract(y_mat, full_img_cell))/y_mat)*100

    from skimage.metrics import structural_similarity as ssim

    rsqr_cell=np.round(r2_score(y_cell, res_cell), decimals=4)
    mae_cell=mean_absolute_error(y_cell, res_cell)
    ssi_m = np.round(ssim(y_mat, full_img_cell, data_range=full_img_cell.max() - full_img_cell.min()), decimals=4)

    rsqrs.append(rsqr_cell)
    maes.append(mae_cell)
    ssims.append(ssi_m)
np.savetxt(rf'c:\Users\mcano\Code\exports\pres240531\{name}_rsqr.csv', rsqrs)
np.savetxt(rf'c:\Users\mcano\Code\exports\pres240531\{name}_mae.csv', maes)
np.savetxt(rf'c:\Users\mcano\Code\exports\pres240531\{name}_ssim.csv', ssims)
# %%
export=0

filename=rf'{name}_linear'
folder=rf'd:\Exports\pres240510'



if export:
    img_export(filename, folder, full_img_roi, full_cell, topography_file, nu_dim)
    histogram_export(full_cell, filename, folder)


export_ref=0

filename=rf'{name}_sim'
folder=rf'd:\Exports\pres240510'

if export_ref: 
    img_export(filename, folder, y_mat, y_cell, topography_file, nu_dim)
    histogram_export(y_cell, filename, folder)


# %%



