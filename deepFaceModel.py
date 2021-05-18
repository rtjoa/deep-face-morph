import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
import os
import cv2
import json
import random

# Loads images and their names (truncates file extension)
def loadImages(directory):
    rawImages = []
    names = []
    
    for filename in os.listdir(directory):
        im = cv2.imread(directory + filename)
        if im is None:
            print(f"Ignoring {filename}, could not read as image.")
        else:
            if rawImages and im.shape != rawImages[0].shape:
                print(f"Ignoring {filename}, as it has a different shape than first image read in.")
            else:
                rawImages.append(im)
                names.append('.'.join(filename.split('.')[:-1]))
    
    return np.stack(rawImages), names

# Perform PCA on data, returns transformed data, eigen basis, and eigen values
def pca(X):
    Sigma = 1/len(X) * np.transpose(X) @ X
    U, S, V = np.linalg.svd(Sigma)
    Z = np.asarray([np.transpose(U) @ x for x in X])
    return Z, U, V

class DeepFaceModel:
    def __init__(self):
        self.built = False
        self.trained = False
    
    # Create untrained encoder and decoder from model params
    def build(self, inputShape, latentDim, filterIncreasePerConv, convolutions, stride, kernelSize):   
        
        # Create encoder model
        inputs = keras.Input(shape = inputShape)
        x = inputs
        for i in range(convolutions):
            x = Conv2D(filters=(i+1)*filterIncreasePerConv, kernel_size=kernelSize, strides=stride, padding='same', activation='relu')(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        self.encoder = keras.Model(inputs = inputs, outputs = latent, name='encoder')
        
        #Create decoder model
        decoderInputs = keras.Input(shape = (latentDim,))
        decoderReshapeDims = np.array(self.encoder.layers[-2].input.shape[1:])
        x = Dense(np.prod(decoderReshapeDims))(decoderInputs)
        x = Reshape(decoderReshapeDims)(x)
        for i in range(convolutions,1,-1):
            x = Conv2DTranspose((i-1)*filterIncreasePerConv, kernelSize, stride, 'same', activation="relu")(x)
        x = Conv2DTranspose(3, kernelSize, stride, 'same', activation='relu')(x)
        self.decoder = keras.Model(inputs=decoderInputs, outputs=x, name='decoder')
        
        self.assembleAutoencoder()
    
    # Combine encoder/decoder into autoencoder
    def assembleAutoencoder(self):
        self.autoencoder = keras.Model(name='autoencoder', inputs = self.encoder.input, outputs = self.decoder(self.encoder(self.encoder.input)))
        self.autoencoder.compile(optimizer='adam',
            loss='mse',
            metrics=['accuracy'])
        self.built = True
    
    # Train model from directory of images
    def train(self, dataFolder, epochs, split_validation=False):
        assert self.built, "Model not built"
        rawImages, self.labels = loadImages(dataFolder)
        if split_validation:
            n_validation = len(rawImages) // 10
            random.shuffle(rawImages)
            trainImages = rawImages[n_validation:]
            valImages = rawImages[:n_validation]
            self.autoencoder.fit(x=trainImages, 
                y=trainImages,
                validation_data=(valImages, valImages),
                epochs=epochs)
        else:
            self.autoencoder.fit(x=rawImages, 
                y=rawImages,
                epochs=epochs)
        latents = self.encoder.predict(rawImages)
        self.transformedLatents, self.eigenBasis, self.eigenValues = pca(latents)
        self.trained = True
    
    # Reconstruct a visualizable image from a transformed, latent vector
    def transformedLatentToImg(self, z):
        assert self.train, "Model not trained"
        latent = self.eigenBasis @ z
        decoded = self.decoder.predict(np.array([latent]))[0]
        return decoded[:,:,::-1].clip(0,255).astype(np.uint8)
    
    # Sort people as (name, trasformed latent vector) tuples
    # by similarity to provided transformed latent vector
    def lookalikes(self, z):
        assert self.train, "Model not trained"
        people = list(zip(self.labels, self.transformedLatents))
        people = [{'name': p[0], 'z': p[1]} for p in people]
        people.sort(key=lambda p: np.square(z - p['z']).sum())
        return people
    
    # Return top lookalike to provided transformed latent vector
    def lookalike(self, z):
        return self.lookalikes(z)[0]
    
    # Save model to directory
    def save(self, directory):
        assert self.train, "Model not trained"
        self.encoder.save(directory + '\\encoder')
        self.decoder.save(directory + '\\decoder')
        np.savetxt(directory + '\\transformedLatents', self.transformedLatents)
        np.savetxt(directory + '\\eigenValues', self.eigenValues)
        np.savetxt(directory + '\\eigenBasis', self.eigenBasis)
        with open(directory + '/labels','w') as f:
            f.write(json.dumps(self.labels))
    
    # Load full trained model from directory
    @staticmethod
    def load(directory):
        dfm = DeepFaceModel()
        dfm.loadModel(directory)
        dfm.transformedLatents = np.loadtxt(directory + '/transformedLatents')
        dfm.eigenValues = np.loadtxt(directory + '/eigenValues')
        dfm.eigenBasis = np.loadtxt(directory + '/eigenBasis')
        with open(directory + '/labels','r') as f:
            dfm.labels = json.loads(f.read())
        dfm.trained = True
        return dfm
    
    # Load a trained autoencoder model from a directory
    def loadModel(self, path):
        self.encoder = keras.models.load_model(path + '/encoder')
        self.decoder = keras.models.load_model(path + '/decoder')
        self.assembleAutoencoder()