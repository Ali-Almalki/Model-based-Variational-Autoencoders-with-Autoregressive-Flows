import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# import tenosrflow probability
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

BATCH_SIZE = 50
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5



class InverseAutoregressiveTransform(tfp.layers.DistributionLambda):

  def __init__(self, made, bijector_num=3,  **kwargs):
    super(InverseAutoregressiveTransform, self).__init__(self._transform, **kwargs)

    if made.params != 2:
      raise ValueError('Argument made must output 2 parameters per input, '
                       'found {}.'.format(made.params))

    self._made = made
    self._bijector_num = bijector_num

  def build(self, input_shape):
    tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=input_shape[1:], dtype=self.dtype),
        self._made
    ])
    super(InverseAutoregressiveTransform, self).build(input_shape)

  def _transform(self, distribution):
    bijectors = list()
    for i in range(self._bijector_num):
      iaf_bijector = tfb.Invert(tfb.MaskedAutoregressiveFlow(lambda x: tf.unstack(self._made(x), axis=-1), is_constant_jacobian=True))
      bijectors.append(iaf_bijector)
    flow_bijector = tfb.Chain(bijectors)
    return tfd.TransformedDistribution(
        bijector=flow_bijector,
        distribution=distribution
    )


class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon


class VAEModel(Model):
    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
           # mu, log_var, z = self.encoder(data)
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction), axis = [1,2,3]
            )
            reconstruction_loss *= self.r_loss_factor
            total_loss = reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss
        }

    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)



class VAE():
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder_ = self.models[1]
        self.encoder = self.encoder_model()
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE


    def encoder_model(self):
        # create model for encode data
        layer = self.encoder_.get_layer('z_params')
        return Model(self.encoder_.inputs, [self.encoder_.layers[-3].output,
                                    self.encoder_.layers[-2].output,
                                    self.encoder_.output])


    def _build(self):
        # define prior distribution for the code, which is an isotropic Gaussian
        prior = tfd.Independent(tfd.Normal(loc=Z_DIM, scale=1.))
        
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0], name='conv_layer_1')(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0], name='conv_layer_2')(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0], name='conv_layer_3')(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0], name='conv_layer_4')(vae_c3)

        vae_z_in = Flatten()(vae_c4)


        t = Dense(tfp.layers.IndependentNormal.params_size(Z_DIM), 
                                 activation=None, name='z_params')(vae_z_in)

        # Add KL divergence regularization loss.
        dist_ = tfp.layers.IndependentNormal(Z_DIM, 
            convert_to_tensor_fn=tfd.Distribution.sample, 
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=KL_TOLERANCE),
            name='z_layer')(t)
                                
        z = InverseAutoregressiveTransform(tfb.AutoregressiveNetwork(
                    params=2,event_shape=Z_DIM, hidden_units=[256, 256],
                    activation='sigmoid'), bijector_num=3)(dist_)
        
        vae_z = z.sample()

        #### DECODER: 
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        vae_dense = Dense(1024, name='dense_layer')(vae_z_input)
        vae_unflatten = Reshape((1,1,DENSE_SIZE), name='unflatten')(vae_dense)
        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0], name='deconv_layer_1')(vae_unflatten)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1], name='deconv_layer_2')(vae_d1)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2], name='deconv_layer_3')(vae_d2)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3], name='deconv_layer_4')(vae_d3)
        

        #### MODELS

    
        vae_encoder = Model(vae_x, vae_z, name = 'encoder')
        vae_decoder = Model(vae_z_input, vae_d4, name = 'decoder')

        vae_full = VAEModel(vae_encoder, vae_decoder, 10000)

        opti = Adam(lr=LEARNING_RATE)
        vae_full.compile(optimizer=opti)
        
        return (vae_full,vae_encoder, vae_decoder)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):

        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE)
        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
