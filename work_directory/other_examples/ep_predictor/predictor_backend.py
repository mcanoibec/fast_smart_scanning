import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from scipy.stats import norm
import seaborn as sns
import math
from nn_backend.modified_loss_functions import cube_msle
from nn_backend.data_import import matrix_cut, data_augment, mask_from_topo
from nn_backend.nn_for_ep import nn_for_ep
from nn_backend.import_xyz_img import import_xyz_img
from nn_backend.conv_imgs import conv_imgs
from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
from nn_backend.extras import simulation_table_import
from skimage.metrics import structural_similarity as ssim

class fc_input:
    def __init__(self, topography_file, curves_file, ref_file, cutout_size):
        self.topography_path = topography_file
        self.curves_path = curves_file
        self.y_path = ref_file
        T_mat, x, y2 = import_xyz_img(self.topography_path)
        self.t_mat=T_mat
        self.og_dim=len(T_mat)
        y_mat, ax_ep, ay_ep= import_xyz_img(self.y_path)
        self.y_mat=y_mat
        nu_dim=min(len(T_mat)-cutout_size+1, len(y_mat))
        self.nu_dim=nu_dim
        self.cutout_size=cutout_size

    def get_dcdz(self, maxlen, curve_height_setpoint, curve_nsamples, curves_max_limit, use_lifts):
        zs=np.round((np.logspace(start=0,stop=math.log10(maxlen),num=curve_nsamples,endpoint=False)+curve_height_setpoint)).astype(int)
        samples = []
        [samples.append(x) for x in zs if x not in samples]
        self.curves_max_limit=curves_max_limit
        if use_lifts:
            if use_lifts:
                og_dim=self.og_dim
                nu_dim=self.nu_dim
                zetas=np.arange(curves_max_limit)
                lifts = [f for f in listdir(self.curves_path) if isfile(join(self.curves_path, f))]
                lift_heights = [int(re.search(r'\d+', filename).group()) for filename in lifts]
                lift_heights.sort()
                lift_samples=np.zeros((og_dim,og_dim,1))
                for l in lift_heights:
                    file, ax_ep, ay_ep= import_xyz_img(rf'{self.curves_path}\LiftPower_{l}_.txt')
                    file=np.expand_dims(file,axis=-1)
                    lift_samples=np.concatenate((lift_samples,file), axis=-1)
                lift_samples=lift_samples[:,:,1:]

                curves_sampled=matrix_cut(lift_samples, nu_dim)
                curves_sampled=curves_sampled.reshape(nu_dim*nu_dim,len(lifts))
                #curves_sampled=curves_sampled[:3]
                b=np.zeros((curves_max_limit,1))

                zzz=np.array(lift_heights)
                projection = np.full((len(curves_sampled),len(zetas)), np.nan)
                for h in tqdm(np.arange(len(curves_sampled))):
                    ccc=curves_sampled[h,:]
                    indices = np.where(np.isin(zetas, zzz.astype(int)))[0]
                    projection[h, indices] = ccc[np.isin(np.round(zzz), zetas)]
                projection=pd.DataFrame(projection)
                projection=projection.interpolate(axis=1)
                projection=projection.bfill(axis=1)
                curves_sampled=np.array(projection).reshape(nu_dim,nu_dim,curves_max_limit)
                curves_sampled=curves_sampled[:,:,samples]

                
        else:
            curves_projected=pd.DataFrame(np.load(rf'{self.curves_path}'))
            curves_projected=curves_projected.interpolate(axis=1)
            curves_projected=curves_projected.bfill(axis=1)

            curves=(np.array(curves_projected)).reshape(128,128,len(curves_projected.iloc[0,:]))
            curves=matrix_cut(curves, self.nu_dim)
            curves_sampled=curves[:,:,samples]
        self.dcdz_sampled=curves_sampled
        self.sample_heights=samples


    def get_vectors(self):
        self.t_mat=matrix_cut(self.t_mat, self.nu_dim)
        self.y_mat=matrix_cut(self.y_mat,self.nu_dim)
        
        self.t_mat=np.expand_dims(self.t_mat, axis=2)
        self.topography=self.t_mat.reshape(self.nu_dim**2)

        self.y=pd.DataFrame(np.reshape(self.y_mat,self.nu_dim**2))

    def get_x(self, maxlen, curve_height_setpoint, curve_nsamples, curves_max_limit, see_data, use_lifts):
        
        self.get_dcdz(maxlen, curve_height_setpoint, curve_nsamples, curves_max_limit, use_lifts)
        self.get_vectors()
        X_mat=np.concatenate((self.t_mat, self.dcdz_sampled), axis=2)

        height_labels=["Z = "+str(x)+" nm" for x in self.sample_heights]
        inputs=["Topography"]
        inputs=np.append(inputs, height_labels)
        
        X=np.reshape(X_mat,(self.nu_dim**2,len(X_mat[0,0,:])))
        X=pd.DataFrame(X, columns=inputs)
        self.x=X
        self.x_mat=X_mat
        if see_data:
            max_v=(np.min(X_mat[:,:,1])+np.max(X_mat[:,:,1]))/2+np.max(X_mat[:,:,1])/2
            min_v=(np.min(X_mat[:,:,1])+np.max(X_mat[:,:,1]))/2-np.max(X_mat[:,:,1])/2
            fig=plt.figure(figsize=(12,10))
            ax=fig.add_subplot(5,3,1)
            pcm=ax.imshow(X_mat[:,:,0], cmap='hot')
            ax.set_title("Topography")
            plt.colorbar(pcm, label="nm")
            ax.axis('off')
            for i in range(2,len(X_mat[0,0,:])+1):
                ax=fig.add_subplot(5,3,i)
                pcm=ax.imshow(X_mat[:,:,i-1], cmap='hot', vmin=min_v, vmax=max_v)
                h=height_labels[i-2]
                ax.set_title("Lift; "+str(h))
                plt.colorbar(pcm, label="$dC/dz$")
                ax.axis('off')


    def apply_masks(self, mask_indexes, mask_from_topo_indexes):
        if hasattr(self, 'x'):
            self.y_masked=pd.DataFrame(self.y.iloc[mask_indexes])
            self.x_masked=pd.DataFrame(self.x.iloc[mask_indexes])
            self.y_masked_with_topo=pd.DataFrame(self.y.iloc[mask_from_topo_indexes])
            self.x_masked_with_topo=pd.DataFrame(self.x.iloc[mask_from_topo_indexes])
        else:
            print("Error: X matrix not found. Use get_x() to obtain")

    def normalize(self):
        if hasattr(self, 'x'):
            curve_scale=StandardScaler()
            curve_scale.fit(self.x_masked)
            self.x_masked_norm=pd.DataFrame(curve_scale.transform(self.x_masked), columns=self.x.columns)
            self.x_masked_with_topo_norm=pd.DataFrame(curve_scale.transform(self.x_masked_with_topo), columns=self.x.columns)
            self.x_norm=pd.DataFrame(curve_scale.transform(self.x), columns=self.x.columns)
        else:
            print("Error: X matrix not found. Use get_x() to obtain")

class conv_input:
    def __init__(self, topography_file, curves_file, ref_file, cutout_size, normalize=1):
        self.topography_path = topography_file
        self.curves_path = curves_file
        self.y_path = ref_file
        T_mat, x, y2 = import_xyz_img(self.topography_path)
        self.t_mat=T_mat
        t=T_mat.reshape(len(T_mat)**2,1)
        y_mat, ax_ep, ay_ep= import_xyz_img(self.y_path)
        self.y_mat=y_mat
        nu_dim=min(len(T_mat)-cutout_size+1, len(y_mat))
        self.nu_dim=nu_dim
        self.cutout_size=cutout_size
        
        if normalize:
            topo_scale=StandardScaler()
            topo_scale.fit(t)
            topo_n=topo_scale.transform(t)
            self.topography_norm=topo_n.reshape(len(T_mat),len(T_mat),1)

    def make_cutouts(self):
        cutout_size=self.cutout_size
        Cv_mat=conv_imgs(self.topography_norm[:,:,0], cutout_size)
        Cv_mat=Cv_mat.reshape(int(math.sqrt(len(Cv_mat))),int(math.sqrt(len(Cv_mat))),cutout_size,cutout_size)
        Cv_mat=matrix_cut(Cv_mat, self.nu_dim)
        Cv_mat=Cv_mat.reshape(self.nu_dim**2,cutout_size,cutout_size)
        Cv_mat=np.expand_dims(Cv_mat, axis=-1)
        self.x_cutouts=Cv_mat


    def make_cutouts_masked(self, mask_indexes, mask_from_topo_indexes):
        nu_dim=self.nu_dim
        cutout_size=self.cutout_size

        y_coord, x_coord= np.unravel_index(mask_indexes, (nu_dim, nu_dim))
        index_1=np.transpose(np.column_stack((y_coord, x_coord)))
        index_1[0]=index_1[0]+int((cutout_size-1)/2)
        index_1[1]=index_1[1]+int((cutout_size-1)/2)
        Cv_mat_masked=conv_imgs(self.topography_norm[:,:,0], cutout_size, index_1)
        Cv_mat_masked=np.expand_dims(Cv_mat_masked, axis=-1)
        self.x_cutouts_masked=Cv_mat_masked

        y_coord, x_coord= np.unravel_index(mask_from_topo_indexes, (nu_dim, nu_dim))
        index_2=np.transpose(np.column_stack((y_coord, x_coord)))
        index_2[0]=index_2[0]+int((cutout_size-1)/2)
        index_2[1]=index_2[1]+int((cutout_size-1)/2)
        Cv_mat_masked_from_topo=conv_imgs(self.topography_norm[:,:,0], cutout_size, index_2)
        Cv_mat_masked_from_topo=np.expand_dims(Cv_mat_masked_from_topo, axis=-1)
        self.x_cutouts_masked_from_topo=Cv_mat_masked_from_topo



    def apply_masks(self, mask_indexes, mask_from_topo_indexes):
        Cv_mat_masked=self.x_cutouts[mask_indexes,:,:,:]
        Cv_mat_masked_from_topo=self.x_cutouts[mask_from_topo_indexes,:,:,:]
        self.x_cutouts_masked=Cv_mat_masked
        self.x_cutouts_masked_from_topo=Cv_mat_masked_from_topo

class sim_tables:
    def __init__(self, fc, conv, sim_tables_folder):
        self.fc=fc
        og_dim=fc.og_dim
        nu_dim=fc.nu_dim
        tab=simulation_table_import(sim_tables_folder, fc.curves_max_limit)
        nu_xs=[]
        nu_ys=[]
        nu_idxs=[]
        for i in np.arange(len(tab)):
            tab_mat=np.zeros((og_dim,og_dim))
            iy=int(tab.at[i,'y'])
            ix=int(tab.at[i,'x'])
            tab_mat[iy,ix]=1
            tab_mat=np.rot90(tab_mat,k=3)
            tab_mat=matrix_cut(tab_mat,nu_dim)
            nu_y, nu_x = np.where(tab_mat == 1)
            nu_idx= np.ravel_multi_index((nu_y,nu_x),(nu_dim,nu_dim))
            
            nu_xs.append(nu_x[0])
            nu_ys.append(nu_y[0])
            nu_idxs.append(nu_idx[0])
        tab.insert(0,'nu x', nu_xs)
        tab.insert(0,'nu y', nu_ys)
        tab.insert(0,'nu idx', nu_idxs)
        y_tab=pd.DataFrame(tab['ep'])
        tab_index=tab['nu idx']
        tab_curves_sampled=tab.iloc[:,6:]
        tab_curves_sampled=tab_curves_sampled.iloc[:,fc.sample_heights]
        tab_T=fc.topography[tab_index].reshape(len(tab_index),1)
        X_tab=np.concatenate((tab_T, tab_curves_sampled), axis=1)
        X_tab=pd.DataFrame(X_tab, columns=self.fc.x.columns)
        self.x_tab=X_tab
        self.y_tab=y_tab
        self.tab_indexes=tab_index
        self.conv=conv
        topo_n_mat=self.conv.topography_norm
        cutout_size=self.conv.cutout_size
        y_coord, x_coord= np.unravel_index(tab_index, (nu_dim, nu_dim))
        tab_index_2=np.transpose(np.column_stack((y_coord, x_coord)))
        tab_index_2[0]=tab_index_2[0]+int((cutout_size-1)/2)
        tab_index_2[1]=tab_index_2[1]+int((cutout_size-1)/2)
        Cv_tab_mat=conv_imgs(topo_n_mat[:,:,0], cutout_size, tab_index_2)
        Cv_tab_mat=np.expand_dims(Cv_tab_mat, axis=-1)
        self.x_cutouts_tab=Cv_tab_mat

    def normalize(self):
        if hasattr(self, 'x_tab'):
            curve_scale=StandardScaler()
            curve_scale.fit(self.fc.x_masked)
            self.x_tab_norm=pd.DataFrame(curve_scale.transform(self.x_tab), columns=self.fc.x.columns)
        else:
            print("Error: X matrix not found. Use get_x() to obtain")

class preprocessing:
    def __init__(self, topography_file, curves_file, ref_file, cutout_size):
        self.topography_path = topography_file
        self.curves_path = curves_file
        self.y_path = ref_file
        T_mat, x, y2 = import_xyz_img(self.topography_path)
        self.t_mat=T_mat
        t=T_mat.reshape(len(T_mat)**2,1)
        y_mat, ax_ep, ay_ep= import_xyz_img(self.y_path)
        self.y_mat=y_mat
        nu_dim=min(len(T_mat)-cutout_size+1, len(y_mat))
        self.nu_dim=nu_dim
        self.t_mat=matrix_cut(self.t_mat, self.nu_dim)
        self.y_mat=matrix_cut(self.y_mat,self.nu_dim)
        
        self.t_mat=np.expand_dims(self.t_mat, axis=2)
        self.topography=self.t_mat.reshape(self.nu_dim**2)
        self.cutout_size=cutout_size
        self.y=pd.DataFrame(np.reshape(self.y_mat,self.nu_dim**2))
    def get_mask(self, binary_threshold, object_size, hole_size, m_ep_thresh):
        bin=mask_from_topo(self.topography,binary_threshold,object_size,hole_size)
        cell=bin.reshape(self.nu_dim**2,1)
        roi_index=pd.DataFrame(copy(cell))
        roi_index=roi_index.index[(roi_index[0])]
        self.mask_from_topo_indexes=roi_index

        mask_ep=pd.DataFrame(copy(self.y))
        th=float(mask_ep.quantile(m_ep_thresh))

        mask_ep[(mask_ep[0]<=th)]=0
        mask_ep[(mask_ep[0]!=0)]=1
        mask_ep=cell*mask_ep
        mask_ep_im=np.reshape(np.array(mask_ep), ((self.nu_dim),(self.nu_dim)))
        cell_index=mask_ep.index[mask_ep[0]>0].tolist()
        self.mask_indexes=cell_index

        fig=plt.figure(figsize=(10,7))
        ax=fig.add_subplot(1,3,1)
        ax.imshow(self.t_mat, cmap='hot')
        ax.set_title('Topography')
        ax.axis('off')
        ax=fig.add_subplot(1,3,2)
        ax.imshow(bin, cmap='gray')
        ax.set_title('Mask from Topography')
        ax.axis('off')
        ax=fig.add_subplot(1,3,3)
        ax.imshow(mask_ep_im)
        ax.axis('off')
        ax.set_title('Final Train Mask')
    def get_fc(self, maxlen=1713, curve_height_setpoint=20, curve_nsamples=12, curves_max_limit=1000, see_data=0, use_lifts=0):
        if hasattr(self, 'mask_indexes'):
            fc = fc_input(self.topography_path, self.curves_path, self.y_path, self.cutout_size)
            fc.get_x(maxlen, curve_height_setpoint, curve_nsamples, curves_max_limit, see_data, use_lifts)
            fc.apply_masks(self.mask_indexes, self.mask_from_topo_indexes)
            fc.normalize()
            self.fc=fc
            self.x_final=self.fc.x_masked_norm
            self.y_final=self.fc.y_masked
        else:
            print("Error: Mask not found. Use get_mask() to obtain")
    def get_conv(self, direct_mask=0):
        if hasattr(self, 'mask_indexes'):
            conv = conv_input(self.topography_path, self.curves_path, self.y_path, self.cutout_size)
            if direct_mask:
                conv.make_cutouts_masked(self.mask_indexes, self.mask_from_topo_indexes)
            else:
                conv.make_cutouts()
                conv.apply_masks(self.mask_indexes, self.mask_from_topo_indexes)
            self.conv=conv
            self.conv_final=self.conv.x_cutouts_masked
        else:
            print("Error: Mask not found. Use get_mask() to obtain")
    def get_tables(self, sim_tables_folder):
        tabs=sim_tables(self.fc, self.conv, sim_tables_folder)
        tabs.normalize()
        self.tabs=tabs
    def add_tables(self):
        self.x_final=pd.DataFrame(np.concatenate((self.fc.x_masked_norm, self.tabs.x_tab_norm), axis=0), columns=self.fc.x.columns)
        self.conv_final=np.concatenate((self.conv.x_cutouts_masked, self.tabs.x_cutouts_tab), axis=0)
        self.y_final=pd.DataFrame(np.concatenate((self.fc.y_masked, self.tabs.y_tab), axis=0))

        train_input_index=np.concatenate((self.mask_indexes,self.tabs.tab_indexes))
        self.train_input_index=train_input_index
class ep_prediction:
    def __init__(self, preproc_data, pctg, augment=1):
        fc, conv, y = preproc_data.x_final,preproc_data.conv_final,preproc_data.y_final
        sss = ShuffleSplit(n_splits=1, test_size=1-(pctg/100), random_state=1)

        sss.get_n_splits(fc, y)
        train_index, test_index = next(sss.split(fc, y))
        train_index=np.sort(train_index)  
        test_index=np.sort(test_index)  



        X_train, X_test = fc.iloc[train_index], fc.iloc[test_index] 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        Cv_train, Cv_test = conv[train_index,:,:], conv[test_index,:,:] 
        if augment:
            Cv_train_aug=data_augment(Cv_train)


            N_aug=int(len(Cv_train_aug[:,0,0,0])/len(Cv_train[:,0,0,0]))

            X_train_aug=copy(X_train)
            y_train_aug=copy(y_train)
            for i in np.arange(N_aug-1):
                X_train_aug=pd.DataFrame(np.concatenate((X_train_aug, X_train), axis=0))
                y_train_aug=pd.DataFrame(np.concatenate((y_train_aug, y_train), axis=0))
            X_train=X_train_aug
            y_train=y_train_aug
            Cv_train=Cv_train_aug
        self.x_train=X_train
        self.y_train=y_train
        self.conv_train=Cv_train
        self.x_test=X_test
        self.y_test=y_test
        self.conv_test=Cv_test
        self.train_index=train_index
        self.test_index=test_index
        self.mask_indexes=preproc_data.mask_indexes
        self.mask_from_topo_indexes=preproc_data.mask_from_topo_indexes
        self.y_masked=preproc_data.fc.y_masked
        self.y=preproc_data.y
        self.nu_dim=preproc_data.nu_dim
        self.pctg=pctg
        self.train_input_index=preproc_data.train_input_index
    def train(self, kernel_dim=3, lossf=cube_msle, batch_sz=200, iterations=300, verb=1, summary=0):
        reg, histo=nn_for_ep(self.x_train,
                     self.conv_train,
                     self.y_train,
                     self.x_test,
                     self.conv_test,
                     self.y_test, 
                     kernel_dim, lossf, batch_sz, iterations, verb, summary)
        self.trained_model=reg
    def predict(self, x, conv):
        res=self.trained_model.predict([x , conv])
        self.prediction=res
    def reconstruct(self):
        nu_dim=self.nu_dim
        roi_index=self.mask_from_topo_indexes
        cell_index=self.mask_indexes
        res=self.prediction
        train_index=self.train_index
        y=self.y
        y_mat=np.array(y).reshape(nu_dim,nu_dim)
        y_cell=self.y_masked
        train_input_index=self.train_input_index

        full_vec_roi=np.zeros(nu_dim**2)+1
        for i in np.arange(len(roi_index)):
            full_vec_roi[roi_index[i]]=res[i]


        for i in y.index[np.array(train_input_index)[train_index]]:
            if i in cell_index and i in np.array(train_input_index)[train_index] :
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

        '''#Blank canvas
        full_vec_roi=np.zeros(nu_dim**2)+1
        #Allocate predicted data 
        for i in np.arange(len(roi_index)):
            full_vec_roi[roi_index[i]]=res[i]

        #Allocate train data directly from y matrix

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

        print(rf'{len(y_cell)}, {len(res_cell)}')
        rsqr_cell=np.round(r2_score(y_cell, res_cell), decimals=4)
        mae_cell=mean_absolute_error(y_cell, res_cell)'''

        error_map=np.subtract(full_img_cell,y_mat)
        error_map_r=(abs(np.subtract(y_mat, full_img_cell))/y_mat)*100

        ssim_idx = np.round(ssim(y_mat, full_img_cell, data_range=full_img_cell.max() - full_img_cell.min()), decimals=4)
        print(ssim)

        rsqr_cell=np.round(r2_score(y_cell, res_cell), decimals=4)
        mae_cell=mean_absolute_error(y_cell, res_cell)

        self.prediction_reconstructed=full_img_roi
        self.error_map=error_map
        self.relative_error_map=error_map_r
        self.rsqr=rsqr_cell
        self.mae=mae_cell
        self.ssim=ssim_idx
        self.prediction_masked_vector=full_cell
        self.res_cell=res_cell
    def display_results(self, min_v=1, max_v=8):
        nu_dim=self.nu_dim
        y=self.y
        y_mat=np.array(y).reshape(nu_dim,nu_dim)
        full_img_roi=self.prediction_reconstructed
        rsqr_cell=self.rsqr
        mae_cell=self.mae
        error_map=self.error_map
        error_map_r=self.relative_error_map
        y_cell=self.y_masked
        full_cell=self.prediction_masked_vector
        res_cell=self.res_cell
        
        fig1=plt.figure(figsize=(14,6))

        ax=fig1.add_subplot(1,2,1)
        pcm=ax.imshow(y_mat, vmin=min_v, vmax=max_v)
        ax.axis("off")
        plt.colorbar(pcm, shrink=0.8, label="$ε_{p}$")
        ax.set_title("Mapa de $ε_p$ original")


        ax=fig1.add_subplot(1,2,2)
        pcm=ax.imshow(full_img_roi, vmin=min_v, vmax=max_v)
        ax.axis("off")
        plt.colorbar(pcm, shrink=0.8, label="$ε_{p}$")
        ax.set_title("Mapa de $ε_p$ predicho")
        ax.text(5, 90, "$R^2 = $"+str(rsqr_cell)+"\n\nError medio\nabsoluto = \n"+str(np.round(mae_cell, decimals=4)), fontsize=11, color="#fff")

        fig1.suptitle("Resultado de red neuronal (train data="+str(self.pctg)+"%)")

        fig2=plt.figure(figsize=(14,6))

        ax=fig2.add_subplot(1,2,1)
        pcm=ax.imshow(error_map, cmap="RdBu", vmin=-3, vmax=3)
        ax.set_title("Mapa de error absoluto")
        ax.axis("off")
        plt.colorbar(pcm, shrink=0.8, label="$ε_p$")
        ax.text(5, 90, "$R^2 = $"+str(rsqr_cell)+"\n\nError medio\nabsoluto = \n"+str(np.round(mae_cell, decimals=4)), fontsize=11, color="#000")

        ax=fig2.add_subplot(1,2,2)
        pcm=ax.imshow(error_map_r, cmap="magma", vmin=0, vmax=50)
        ax.set_title("Mapa de error relativo")
        ax.axis("off")
        plt.colorbar(pcm, shrink=0.8, label="%")


        fig2.suptitle("Resultado de red neuronal (train data="+str(self.pctg)+"%)")

        fig3=plt.figure(figsize=(14,6))

        ax=fig3.add_subplot(1,2,1)
        binss=(np.arange(4*max_v)*0.25)+0.25+min_v
        xaxis=(np.arange(9))
        ax.hist(y_cell, histtype="step", color="#000",bins=binss)
        ax.hist(full_cell, histtype="bar", color="#92def7",bins=binss)
        ax.set_xticks(xaxis)
        ax.set_xlim([min_v,max_v])
        ax.set_ylim([0,1000])
        ax.set_title("Distribución de datos")
        ax.legend(["Original","Predicción \n(train data="+str(self.pctg)+"%)"])
        ax.set_xlabel("$ε_{p}$")
        ax.set_ylabel("Count")
        ax.text(1.2, 750, "$R^2 = $"+str(rsqr_cell)+"\nError medio absoluto = "+str(np.round(mae_cell, decimals=4)), fontsize=11, color="#000")



        thresh=1
        print("The error map presents "+str(len(error_map[abs(error_map)>thresh]))+" error values > "+str(thresh)+" ("+str(np.round((len(error_map[abs(error_map)>thresh])/len(y_cell))*100, decimals=2))+"%)")
        plt.figure()
        err=pd.DataFrame(np.subtract(res_cell,y_cell))
        ax=fig3.add_subplot(1,2,2)
        ax = sns.kdeplot(data=err, ax=ax)
        legend=ax.legend()
        legend.remove()
        ax.set_title("Distribución de error en datos predichos")
        ax.set_xlabel("Error")
        ax.set_ylabel("Densidad de probabilidad")
        ax.set_xlim((-3,3))
        ax.set_ylim((0,3))
        ax.grid('on', alpha=0.5)
        err_mean, err_std=norm.fit(err)
        ax.text(-2.8, 1.25, "Centro = "+str(np.round(err_mean, decimals=4))+"\nDesv. Estándar  = "+str(np.round(err_std, decimals=4)), fontsize=11, color="#000")

        fig3.suptitle("Resultado de red neuronal (train data="+str(self.pctg)+"%)")
