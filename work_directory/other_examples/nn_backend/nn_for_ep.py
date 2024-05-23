#Neural Network Functions
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, concatenate
from keras.optimizers import Adam
from keras.utils import set_random_seed
from nn_backend.modified_loss_functions import cube_msle


def nn_for_ep(xtrain,convtrain,ytrain,xtest,convtest,ytest, kernel_dim=3, lossf=cube_msle, batch_sz=200, iterations=300, verb=1, summary=0):
    set_random_seed(1)
    inp_x=Input(shape=(len(xtrain.columns),))
    inp_cv=Input(shape=(np.shape(convtrain[0])))


    cv=Conv2D(filters=32, kernel_size=(kernel_dim, kernel_dim), activation='relu')(inp_cv)
    cv=Flatten()(cv)
    cv=Dense(256,activation ="relu")(cv)
    cv=Model(inputs=inp_cv, outputs=cv)

    x=Dense(256, activation='relu')(inp_x)
    x=Model(inputs=inp_x, outputs=x)

    mix=concatenate([x.output,cv.output])

    mix=Dense(256, activation='relu')(mix)
    mix=Dense(128, activation='relu')(mix)
    mix=Dense(64, activation='relu')(mix)
    mix=Dense(32, activation='relu')(mix)
    out=Dense(1)(mix)

    reg = Model(inputs=[x.input, cv.input], outputs=out)
    if summary:
        reg.summary()
    reg.compile(loss=lossf,optimizer=Adam(),metrics=['mae'])
    histo=reg.fit([xtrain,convtrain], ytrain, batch_size=batch_sz, epochs=iterations, validation_data=[[xtest,convtest],ytest], verbose=verb)
    return reg, histo