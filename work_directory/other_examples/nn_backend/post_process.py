#Neural Network Functions
import numpy as np
from copy import copy

from nn_backend.data_import import matrix_cut


def prediction_reconstruction_with_train(res, ref, cell_index, train_index, test_index, nu_dim):
    y_index_train=np.array(cell_index)[train_index]
    y_index_test=np.array(cell_index)[test_index]

    train_img=np.zeros(nu_dim**2)+1
    for i in y_index_train:
        train_img[i]=ref.iloc[i,0]
    train_img=np.reshape(train_img,(nu_dim,nu_dim))

    #VISUALIZACIÓN DE DATOS CALCULADOS
    test_img=np.zeros(nu_dim**2)+1
    for i in np.arange(len(y_index_test)):
        test_img[y_index_test[i]]=res[i]


    #IMAGEN COMPLETA, CÁLCULO+ORIGINAL
    full_vec=copy(test_img)
    for i in y_index_train:
        full_vec[i]=ref.iloc[i,0]
    return full_vec


def img_export(filename, folder, matrix, vector, topography_file, nu_dim, og_dim):
    Original_file=copy(topography_file)
    Image_to_export=matrix
    ########
    template=np.loadtxt(rf'{Original_file}', skiprows=4)
    template=np.reshape(template, (og_dim,og_dim,3))
    template=matrix_cut(template, nu_dim)
    template=template.reshape((nu_dim)**2,3)
    new_xyz=copy(template)
    flipped_img=np.flipud(Image_to_export)
    flipped_img=flipped_img.reshape(len(Image_to_export)**2)
    new_xyz[:,2]=flipped_img
    np.savetxt('temp_xyz_data.txt',new_xyz, delimiter='\t')


    header='WSxM file copyright UAM\nWSxM ASCII XYZ file\nX[nm]\tY[nm]\tZ[ep]\n'
    with open('temp_xyz_data.txt') as fp:
        data = fp.read()

    full_export=copy(header)
    full_export+='\n'
    full_export+=data

    with open (rf'{folder}\{filename}.txt', 'w') as fp:
        fp.write(full_export)

    np.savetxt(rf'{folder}\{filename}_vec.csv',vector)   

def histogram_export(input, filename, folder, start=0, stop=8, step=0.1):
    hist, bins= np.histogram(input, np.arange(start,stop,step))
    hist=np.column_stack((bins[1:], hist))
    np.savetxt(rf'{folder}\{filename}_hist.csv', hist, delimiter=',')