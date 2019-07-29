import keras
from keras.utils import plot_model
from unet import UNet

net = UNet()
model = net.model
model.compile(optimizer = keras.optimizers.Adam(lr=0.001), loss = 'categorical_crossentropy')
model.summary()
plot_model(model, to_file='model.png')
