import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, \
                         Cropping2D, Conv2DTranspose, UpSampling2D, Dropout
from keras.models import Model

class UNet:
  def __init__(self, shape=(572, 572, 1), filters=64, kernel_size=3, convs_num=2, BN=False, activation='relu', level=5):
      in_img = Input(shape=shape, dtype='float')

      # Contracting Path
      conv = in_img
      skips = []
      for l in range(1, level+1):
          name = 'dn' + str(l)
          conv = self.convs(conv, filters, kernel_size, convs_num, BN, activation, name)
          if l < level:
              skips.append(conv)
              conv = MaxPooling2D(2, 2, name='dn'+str(l+1))(conv)
              filters *= 2

      # Expansive Path
      for l, skip in enumerate(reversed(skips)):
          name = 'up' + str(level-l-1)
          filters //= 2
          skip_size = int(skip.shape[1])
          upsample_size = int(2 * conv.shape[1])
          crop_size = (skip_size - upsample_size) // 2
          if skip_size%2: crop_size = ((crop_size, crop_size+1), (crop_size, crop_size+1))
          skip = Cropping2D(cropping=crop_size, name='crop'+str(level-l-1))(skip)
          conv = Conv2DTranspose(filters, kernel_size=1, strides=2, name=name)(conv)
          conv = Concatenate(name='concat'+str(level-l-1))([skip, conv])
          conv = self.convs(conv, filters, kernel_size, convs_num, BN, activation, name)

      out = Conv2D(2, 1, activation='sigmoid', name='output')(conv)
      self.model = Model(in_img, out)

  def convs(self, conv, filters, kernel_size, convs_num, BN, activation, name):
      data_format = 'channels_last'
      NORM_AXIS = 3

      for i in range(convs_num):
          layer_name = name+'_conv'+str(i)
          conv = Conv2D(filters, kernel_size, activation=activation, name=layer_name)(conv)
          if BN: conv = BatchNormalization(axis=NORM_AXIS, name=name+'_BN'+str(i))(conv)

      return conv
