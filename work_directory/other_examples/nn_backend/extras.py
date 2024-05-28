
import pandas as pd
import numpy as np
from copy import copy
from os import listdir
from os.path import isfile, join


def simulation_table_import(sim_tables_folder, curves_max_limit):
    onlyfiles = [f for f in listdir(sim_tables_folder) if isfile(join(sim_tables_folder, f))]
    zetas=np.arange(curves_max_limit)
    stack=np.zeros((1,curves_max_limit+3))


    for k in np.arange(len(onlyfiles)):
        #EXTRACT


        a=pd.read_table(rf'{sim_tables_folder}\{onlyfiles[k]}', skiprows=19)
        a=a.pivot_table(columns='Z[nm]', values='dCdz[aF/nm]', index='ep[none]')
        eps=np.array(a.index)

        #PROJECTION
        b=np.zeros((curves_max_limit,1))
        for w in np.arange(len(a)):
            zs=np.array(a.columns)
            ccc=a.iloc[w,:]
            projection = np.full((len(zetas)), np.nan)
            indices = np.where(np.isin(zetas, zs.astype(int)))[0]
            projection[indices] = ccc[np.isin(np.round(zs), zetas)]
            projection=np.expand_dims(projection, axis=1)
            b=np.concatenate((b,projection), axis=1)
        b=b.T
        b=b[1:,]
        b=pd.DataFrame(b)
        b=b.interpolate(axis=1)
        b=b.bfill(axis=1)

        #INDEXES
            
        files=onlyfiles[k]
        files=files.replace("Tabla_I_", "")
        files=files.replace("_J_", "_")
        files=files.replace(".txt", "")
        files=np.array(files.split("_")).astype(int)
        files=np.expand_dims(files, axis=1)

        idxs=np.zeros((2,1))
        for u in np.arange(len(eps)):
            idxs=np.concatenate((idxs,files), axis=1)
        idxs=idxs[:,1:]
        idxs_t=copy(idxs).T

        #PACKAGE
        t=np.column_stack((idxs_t,eps,np.array(b)))
        stack=np.concatenate((stack, t), axis=0)

    names=['y','x','ep']
    for i in zetas:
        names.append(rf'Z={i} nm')


    stack=stack[1:]
    stack=pd.DataFrame(stack, columns=names)
    return stack