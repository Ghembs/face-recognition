from keras.models import Sequential, Model
from keras.layers import Flatten, Dropout, Activation, Permute
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from scipy.io import loadmat
import numpy as np

# WARNING : important for images and tensors dimensions ordering
K.set_image_data_format('channels_last')


def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1, bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        # L.append( Convolution2D(cdim, 3, 3, border_mode='same',
        # activation='relu', name=convname) ) # Keras 1
        L.append(Convolution2D(cdim, kernel_size=(3, 3), padding='same',
                               activation='relu', name=convname))  # Keras 2
    
    L.append(MaxPooling2D((2, 2), strides=(2, 2)))
    
    return L


def vgg_face_blank():
    
    withDO = True  # no effect during evaluation but useful for fine-tuning
    
    if True:
        mdl = Sequential()
        
        # First layer is a dummy-permutation = Identity to specify input shape
        mdl.add(Permute((1, 2, 3), input_shape=(224, 224, 3)))  # WARNING : 0 is the sample dim

        for l in convblock(64, 1, bits=2):
            mdl.add(l)

        for l in convblock(128, 2, bits=2):
            mdl.add(l)
        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)
        
        # mdl.add( Convolution2D(4096, 7, 7, activation='relu', name='fc6') ) # Keras 1
        mdl.add(Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6'))  # Keras 2
        if withDO:
            mdl.add(Dropout(0.5))
        # mdl.add( Convolution2D(4096, 1, 1, activation='relu', name='fc7') ) # Keras 1
        mdl.add(Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7'))  # Keras 2
        if withDO:
            mdl.add(Dropout(0.5))
        # mdl.add( Convolution2D(2622, 1, 1, name='fc8') ) # Keras 1
        mdl.add(Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8'))  # Keras 2
        mdl.add(Flatten())
        mdl.add(Activation('softmax'))
        
        return mdl
    
    # else:
        # See following link for a version based on Keras functional API :
        # gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
        # raise ValueError('not implemented')


def copy_mat_to_keras(kmodel, layers):

    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    # prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0, 1, 2, 3)  # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(layers.shape[1]):
        matname = layers[0, i][0, 0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            # print matname
            l_weights = layers[0, i][0, 0].weights[0, 0]
            l_bias = layers[0, i][0, 0].weights[0, 1]
            f_l_weights = l_weights.transpose(prmt)
            # f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
            # f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:, 0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:, 0]])
            # print '------------------------------------------'


def weight_compare(kmodel, layers):
    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    # prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0, 1, 2, 3)  # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(layers.shape[1]):
        matname = layers[0, i][0, 0].name[0]
        mattype = layers[0, i][0, 0].type[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print(matname, mattype)
            print(layers[0, i][0, 0].weights[0, 0].transpose(prmt).shape,
                  layers[0, i][0, 0].weights[0, 1].shape)
            print(kmodel.layers[kindex].get_weights()[0].shape,
                  kmodel.layers[kindex].get_weights()[1].shape)
            print('------------------------------------------')
        else:
            print('MISSING : ', matname, mattype)
            print('------------------------------------------')


def load_net(path):
    data = loadmat(path, matlab_compatible=False, struct_as_record=False)
    if 'vgg-face.mat' in path:
        layers = data['layers']
        description = data['meta'][0, 0].classes[0, 0].description
    else:
        net = data['net'][0, 0]
        layers = net.layers
        description = net.classes[0, 0].description
    print("Layers shape = {}\tDescription shape = {}\n".format(layers.shape, description.shape))
    return layers


def pred(kmodel, crpimg, description, transform=False):
    
    # transform=True seems more robust but I think the RGB channels are not in right order
    
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:, :, 0] -= 129.1863
        imarr[:, :, 1] -= 104.7624
        imarr[:, :, 2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        # aux = copy.copy(imarr)
        # imarr[:, :, 0] = aux[:, :, 2]
        # imarr[:, :, 2] = aux[:, :, 0]

        # imarr[:,:,0] -= 129.1863
        # imarr[:,:,1] -= 104.7624
        # imarr[:,:,2] -= 93.5940

    # imarr = imarr.transpose((2,0,1)) # INFO : for 'th' setting of 'dim_ordering'
    imarr = np.expand_dims(imarr, axis=0)

    out = kmodel.predict(imarr)

    best_index = np.argmax(out, axis=1)[0]
    best_name = description[best_index, 0]
    print(best_index, best_name[0], out[0, best_index], [np.min(out), np.max(out)])


def my_pred(kmodel, crpimg, transform=False):

    imarr = np.array(crpimg).astype(np.float32)
    if transform:
        imarr[:, :, 0] -= 129.1863
        imarr[:, :, 1] -= 104.7624
        imarr[:, :, 2] -= 93.5940

    imarr = np.expand_dims(imarr, axis=0)
    # imarr = preprocess_input(imarr)
    out = kmodel.predict(imarr)
    return out


def get_model(path_to_mat):
    verbose = True
    facemodel = vgg_face_blank()
    if verbose:
        print(facemodel.summary())
    layers = load_net(path_to_mat)
    copy_mat_to_keras(facemodel, layers)
    realmodel = Model(inputs =facemodel.layers[0].input, outputs =facemodel.layers[-2].output)
    return realmodel
