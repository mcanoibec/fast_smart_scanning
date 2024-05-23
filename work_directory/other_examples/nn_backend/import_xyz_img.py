def import_xyz_img(filename,s=4):
    import numpy as np
    import math
    file = np.loadtxt(filename, skiprows=s)
    z = file[:,2]
    axis = int(math.sqrt(len(z)))
    z=np.reshape(z,(axis,axis))

    x = file[:,0]
    x = np.array(list(dict.fromkeys(x)))

    y = file[:,1]
    y = np.array(list(dict.fromkeys(x)))
    return z,x,y