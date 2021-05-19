import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import os
import cv2
import json
import random

# Loads images and their names (truncates file extension)
def load_images(directory):
    raw_images = []
    names = []

    for filename in os.listdir(directory):
        im = cv2.imread(directory + filename)
        if im is None:
            print(f"Ignoring {filename}, could not read as image.")
        else:
            if raw_images and im.shape != raw_images[0].shape:
                print(f"Ignoring {filename}, as it has a different shape than first image read in.")
            else:
                raw_images.append(im)
                names.append('.'.join(filename.split('.')[:-1]))

    return np.stack(raw_images), names

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
    def build(self, input_shape, latent_dim, filter_increase_per_conv, convolutions, stride, kernel_size):
        # Create encoder model
        inputs = keras.Input(shape = input_shape)
        x = inputs
        for i in range(convolutions):
            layer = Conv2D(
                filters=(i+1)*filter_increase_per_conv,
                kernel_size=kernel_size,
                strides=stride,
                padding='same',
                activation='relu'
                )
            x = layer(x)
        x = Flatten()(x)
        latent = Dense(latent_dim)(x)
        self.encoder = keras.Model(inputs=inputs, outputs=latent, name='encoder')

        # Create decoder model
        decoder_inputs = keras.Input(shape = (latent_dim,))
        decoder_reshape_dims = np.array(self.encoder.layers[-2].input.shape[1:])
        x = Dense(np.prod(decoder_reshape_dims))(decoder_inputs)
        x = Reshape(decoder_reshape_dims)(x)
        for i in range(convolutions,1,-1):
            layer = Conv2DTranspose(
                (i-1)*filter_increase_per_conv,
                kernel_size,
                stride,
                'same',
                activation="relu"
                )
            x = layer(x)
        x = Conv2DTranspose(3, kernel_size, stride, 'same', activation='relu')(x)
        self.decoder = keras.Model(inputs=decoder_inputs, outputs=x, name='decoder')

        self.assemble_autoencoder()

    # Combine encoder/decoder into autoencoder
    def assemble_autoencoder(self):
        self.autoencoder = keras.Model(
            name='autoencoder',
            inputs=self.encoder.input,
            outputs=self.decoder(self.encoder(self.encoder.input)))
        self.autoencoder.compile(optimizer='adam',
            loss='mse',
            metrics=['accuracy'])
        self.built = True

    # Train model from directory of images
    def train(self, data_path, epochs, split_validation=False, patience=999):
        assert self.built, "Model not built"
        raw_images, self.labels = load_images(data_path)
        if split_validation:
            n_validation = len(raw_images) // 20
            random.shuffle(raw_images)
            train_images = raw_images[n_validation:]
            val_images = raw_images[:n_validation]

            es = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
                )

            self.autoencoder.fit(x=train_images,
                y=train_images,
                validation_data=(val_images, val_images),
                epochs=epochs,
                callbacks=[es]
                )
        else:
            self.autoencoder.fit(x=raw_images,
                y=raw_images,
                epochs=epochs
                )
        latents = self.encoder.predict(raw_images)
        self.transformed_latents, self.eigenbasis, self.eigenvalues = pca(latents)
        self.trained = True

    # Reconstruct a visualizable image from a transformed, latent vector
    def transformed_latent_to_img(self, z):
        assert self.train, "Model not trained"
        latent = self.eigenbasis @ z
        decoded = self.decoder.predict(np.array([latent]))[0]
        return decoded[:,:,::-1].clip(0,255).astype(np.uint8)

    # Sort people as (name, trasformed latent vector) tuples
    # by similarity to provided transformed latent vector
    def lookalikes(self, z):
        assert self.train, "Model not trained"
        people = list(zip(self.labels, self.transformed_latents))
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
        np.savetxt(directory + '\\transformedLatents', self.transformed_latents)
        np.savetxt(directory + '\\eigenValues', self.eigenvalues)
        np.savetxt(directory + '\\eigenBasis', self.eigenbasis)
        with open(directory + '/labels','w') as f:
            f.write(json.dumps(self.labels))

    # Load full trained model from directory
    @staticmethod
    def load(directory):
        dfm = DeepFaceModel()
        dfm.load_model(directory)
        dfm.transformed_latents = np.loadtxt(directory + '/transformedLatents')
        dfm.eigenvalues = np.loadtxt(directory + '/eigenValues')
        dfm.eigenbasis = np.loadtxt(directory + '/eigenBasis')
        with open(directory + '/labels','r') as f:
            dfm.labels = json.loads(f.read())
        dfm.trained = True
        return dfm

    # Load a trained autoencoder model from a directory
    def load_model(self, path):
        self.encoder = keras.models.load_model(path + '/encoder')
        self.decoder = keras.models.load_model(path + '/decoder')
        self.assemble_autoencoder()
