"""
Image dataset pre-processing and AutoEncoder.

author: yajue
date:20171017
"""

import glob, os, time, chainer, sys
import chainer.optimizers as O
import chainer.links as L
import chainer.functions as F
from chainer import iterators, cuda, serializers, Variable
import numpy as np
from PIL import Image
import random
import chainer.configuration


# In[2]:

class TaskImageDataset(chainer.datasets.ImageDataset):
    """
    Dataset of images built from a list of paths to image files.

    author: yajue
    date: 20171017
    """
    def __init__(self, paths, root, dtype=np.float32, img_resize=(128, 96)):
        """
        Initialization.
        :param paths: A list of paths to image files
        :param root: Root directory
        :param dtype: data type
        :param img_resize: image resize

        author: yajue
        date: 20171017
        """
        super(TaskImageDataset, self).__init__(paths, root, dtype)
        self.img_resize = img_resize
        self.img_len = len(paths)
        # grayscale
#         self.img_mean = np.zeros((96, 128, 3), dtype=self._dtype)
        self.img_mean = np.zeros((96, 128), dtype=self._dtype)
        self.calc_mean_image()

    def read_image_as_array(self, path, dtype):
        """
        Read image as an array.
        :param path: The path to a image file
        :param dtype: data type
        :return:

        author: yajue
        date: 20171017
        """
        f = Image.open(path)

        try:
            f = f.resize(self.img_resize)
            # grayscale
            f = f.convert('L')
            image = np.asarray(f, dtype=dtype)
            image = (image / 128) - 1

        finally:
            if hasattr(f, 'close'):
                f.close()
        return image
    
    def calc_mean_image(self):
        for i in range(0, self.img_len):
            path = os.path.join(self._root, self._paths[i])
            self.img_mean += self.read_image_as_array(path, self._dtype)
        self.img_mean /= self.img_len
        # grayscale
        self.img_mean = self.img_mean[:, :, np.newaxis]

    def get_example(self, i):
        """
        Get example from the dataset.
        :param i: Index of the data
        :return:

        author: yajue
        date: 20171007
        """
        path = os.path.join(self._root, self._paths[i])
        image = self.read_image_as_array(path, self._dtype)

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, np.newaxis]
        return image.transpose(2, 0, 1)


class ImageAENetwork(chainer.Chain):
    """
    AutoEocoder network for images.

    author: yajue
    date: 20171006
    """
    def __init__(self, net_idx):
        super(ImageAENetwork, self).__init__()
        with self.init_scope():
            np.set_printoptions(precision=5, suppress=True)

            self.net_idx = net_idx

            if self.net_idx == 1:
                """
                Network1
                """
                # encode layers
                self.conv1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5, stride=1)
                self.conv2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, stride=1)
                self.conv3 = L.Convolution2D(in_channels=64, out_channels=128, ksize=(7, 5), stride=1)
                self.conv4 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1)
                self.fc5 = L.Linear(None, 50)

                # decode layers
                self.dfc5 = L.Linear(50, 256 * 3 * 3)
                self.dconv4 = L.Deconvolution2D(in_channels=256, out_channels=128, ksize=3, stride=1)
                self.dconv3 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=(7, 5), stride=1)
                self.dconv2 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=5, stride=1)
                self.dconv1 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=5, stride=1)

            elif self.net_idx == 2:
                """
                Network2
                """
                # encode layers
                self.conv1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=7, stride=2, pad=3)
                self.conv2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=5, pad=2)
                self.conv3 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, pad=1)
                self.conv4 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, pad=1)
                self.fc5 = L.Linear(None, 150)
                self.fc6 = L.Linear(None, 50)

                # decode layers
                self.dfc6 = L.Linear(None, 150)
                # self.dfc5 = L.Linear(None, 256 * 5 * 7)
                self.dfc5 = L.Linear(None, 256 * 6 * 8)
                self.dconv4 = L.Deconvolution2D(in_channels=256, out_channels=128, ksize=3, pad=1)
                self.dconv3 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=3, pad=1)
                self.dconv2 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=5, pad=2)
                self.dconv1 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=7, stride=2, pad=3, outsize=(96, 128))

            elif self.net_idx == 3:
                """
                Network3
                """
                # encoder layers
                self.conv1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5, stride=1, pad=2)
                self.conv2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=3, stride=1, pad=1)
                self.conv3 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=1, pad=1)
                self.conv4 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1, pad=1)
                self.fc5 = L.Linear(None, 1000)
                self.bn5 = L.BatchNormalization(1000)
                self.fc6 = L.Linear(None, 50)
                self.bn6 = L.BatchNormalization(50)

                # decoder layers
                self.dfc6 = L.Linear(None, 1000)
                self.dfc5 = L.Linear(None, 256 * 6 * 8)
                self.dconv4 = L.Deconvolution2D(in_channels=256, out_channels=128, ksize=3, stride=1, pad=1)
                self.dconv3 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=3, stride=1, pad=1)
                self.dconv2 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=3, stride=1, pad=1)
                self.dconv1 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=5, stride=1, pad=2)

            elif self.net_idx == 4:
                """
                Network4
                """
                # encoder layers
                self.conv1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5, stride=1, pad=2)
                self.conv2 = L.Convolution2D(in_channels=32, out_channels=16, ksize=5, stride=2, pad=2)
                self.conv3 = L.Convolution2D(in_channels=16, out_channels=16, ksize=5, stride=1, pad=2)
                self.conv4 = L.Convolution2D(in_channels=16, out_channels=8, ksize=5, stride=2, pad=2)
                self.fc5 = L.Linear(None, 100)
                self.fc6 = L.Linear(None, 100)

                # decoder layers
                self.dfc6 = L.Linear(100, 100)
                self.dfc5 = L.Linear(100, 8 * 24 * 32)
                self.dconv4 = L.Deconvolution2D(in_channels=8, out_channels=16, ksize=5, stride=2, pad=2, outsize=(48, 64))
                self.dconv3 = L.Deconvolution2D(in_channels=16, out_channels=16, ksize=5, stride=1, pad=2)
                self.dconv2 = L.Deconvolution2D(in_channels=16, out_channels=32, ksize=5, stride=2, pad=2, outsize=(96, 128))
                self.dconv1 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=5, stride=1, pad=2)
                
            elif self.net_idx == 5:
                """
                Network5
                """
                # encoder layers
                self.conv1 = L.Convolution2D(in_channels=3, out_channels=32, ksize=5, stride=1, pad=2)
                self.conv2 = L.Convolution2D(in_channels=32, out_channels=16, ksize=5, stride=1, pad=2)
                self.fc5 = L.Linear(None, 100)
                self.fc6 = L.Linear(None, 10)
                
                # decoder layers
                self.dfc6 = L.Linear(10, 100)
                self.dfc5 = L.Linear(10, 16 * 96 * 128)
                self.dconv2 = L.Deconvolution2D(in_channels=16, out_channels=32, ksize=5, stride=1, pad=2)
                self.dconv1 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=5, stride=1, pad=2)
                
            elif self.net_idx == 6:
                """
                Network6
                """
                # encoder layers
                # grayscale
                self.conv1 = L.Convolution2D(in_channels=1, out_channels=32, ksize=5, stride=1, pad=2)
                self.conv2 = L.Convolution2D(in_channels=32, out_channels=16, ksize=5, stride=1, pad=2)
                self.fc5 = L.Linear(None, 100)
                self.fc6 = L.Linear(None, 10)
                
                # decoder layers
                self.dfc6 = L.Linear(10, 100)
                self.dfc5 = L.Linear(100, 16 * 96 * 128)
                self.dconv2 = L.Deconvolution2D(in_channels=16, out_channels=32, ksize=5, stride=1, pad=2)
                # grayscale
                self.dconv1 = L.Deconvolution2D(in_channels=32, out_channels=1, ksize=5, stride=1, pad=2)

            else:
                print('ERROR: No network{}'.format(self.net_idx))
                sys.exit()

    def encoder_net1(self, x):
        h = F.relu(self.conv1(x))
        # print('after conv1 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.max_pooling_2d(h, ksize=2)
        # print('after conv1 pooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.relu(self.conv2(h))
        # print('after conv2 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.max_pooling_2d(h, ksize=2)
        # print('after conv2 pooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.relu(self.conv3(h))
        # print('after conv3 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.max_pooling_2d(h, ksize=(3, 5))
        # print('after conv3 pooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.relu(self.conv4(h))
        # print('after conv4 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = self.fc5(h)
        # print('after fc5 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        return h

    def decoder_net1(self, h, input_shape):
        h = F.relu(self.dfc5(h))
        # print('after dfc5 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.relu(self.dconv4(h.reshape((input_shape[0], 256, 3, 3))))
        # print('after dconv4 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.unpooling_2d(h, ksize=(3, 5), outsize=(15, 25))
        # print('after dconv3 unpooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.relu(self.dconv3(h))
        # print('after dconv3 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.unpooling_2d(h, ksize=2, outsize=(42, 58))
        # print('after dconv2 unpooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.relu(self.dconv2(h))
        # print('after dconv2 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        #
        h = F.unpooling_2d(h, ksize=2, outsize=(92, 124))
        # print('after dconv1 unpooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.relu(self.dconv1(h))
        # print('after dconv1 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        print(h)

        return h

    def encoder_net2(self, x):
        h = F.relu(self.conv1(x))
        # print('after conv1 relu shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        # h = F.local_response_normalization(h)
        # print('local normalization h: ', h.data)
        # raw_input('enter...')
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        # print('after conv1 pooling shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.relu(self.conv2(h))
        # print('after conv2 relu shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        # h = F.local_response_normalization(h)
        # print('after conv2 local normalization shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        # print('after conv2 pooling shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.relu(self.conv3(h))
        # print('after conv3 relu shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.relu(self.conv4(h))
        # print('after conv4 relu shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        # print('after conv4 pooling shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.dropout(F.relu(self.fc5(h)))
        # h = F.relu(self.fc5(h))
        # print('after fc5 relu shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = self.fc6(h)
        # print('after fc6 shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        return h

    def decoder_net2(self, h, input_shape):
        # decode
        h = self.dfc6(h)
        # print('after dfc6 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        # h = F.dropout(F.relu(self.dfc5(h)))
        h = F.relu(self.dfc5(h))
        # print('after dfc5 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.unpooling_2d(h.reshape((input_shape[0], 256, 6, 8)), ksize=3, stride=2, outsize=(12, 16))
        # print('after dconv4 unpooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.relu(self.dconv4(h))
        # print('after dconv4 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.relu(self.dconv3(h))
        # print('after dconv3 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.unpooling_2d(h, ksize=3, stride=2)
        # print('after dconv2 unpooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.relu(self.dconv2(h))
        # print('after dconv2 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        h = F.unpooling_2d(h, ksize=3, stride=2, outsize=(48, 64))
        # print('after dconv1 unpooling output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')
        h = F.relu(self.dconv1(h))
        # print('after dconv1 output shape: ', h.data.shape)
        # print('h: ', h.data)
        # raw_input('enter...')

        return h

    def encoder_net3(self, x):
        h = F.relu(self.conv1(x))
        # print('after conv1 output shape: ', h.data.shape)
        h = F.max_pooling_2d(h, ksize=2)
        # print('after conv1 pooling output shape: ', h.data.shape)

        h = F.relu(self.conv2(h))
        # print('after conv2 output shape: ', h.data.shape)
        h = F.max_pooling_2d(h, ksize=2)
        # print('after conv2 pooling output shape: ', h.data.shape)

        h = F.relu(self.conv3(h))
        # print('after conv3 output shape: ', h.data.shape)
        h = F.max_pooling_2d(h, ksize=2)
        # print('after conv3 pooling output shape: ', h.data.shape)

        h = F.relu(self.conv4(h))
        # print('after conv4 output shape: ', h.data.shape)
        h = F.max_pooling_2d(h, ksize=2)
        # print('after conv4 pooling output shape: ', h.data.shape)

        h = self.fc5(h)
        # print('after fc5 output shape: ', h.data.shape)
        h = self.bn5(h)
        # print('after fc5 batch normalization output shape: ', h.data.shape)
        h = F.relu(h)

        h = self.fc6(h)
        h = self.bn6(h)

        return h

    def decoder_net3(self, h, input_shape):
        h = self.dfc6(h)
        h = self.bn5(h)
        h = F.relu(h)
        # print('after dfc6 output shape: ', h.data.shape)

        h = F.relu(self.dfc5(h))
        # print('after dfc5 output shape: ', h.data.shape)

        h = F.unpooling_2d(h.reshape((input_shape[0], 256, 6, 8)), ksize=2, outsize=(12, 16))
        # print('after dconv4 unpooling output shape: ', h.data.shape)
        h = F.relu(self.dconv4(h))
        # print('after dconv4 output shape: ', h.data.shape)

        h = F.unpooling_2d(h, ksize=2, outsize=(24, 32))
        # print('after dconv3 unpooling output shape: ', h.data.shape)
        h = F.relu(self.dconv3(h))
        # print('after dconv3 output shape: ', h.data.shape)

        h = F.unpooling_2d(h, ksize=2, outsize=(48, 64))
        # print('after dconv2 unpooling output shape: ', h.data.shape)
        h = F.relu(self.dconv2(h))
        # print('after dconv2 output shape: ', h.data.shape)

        h = F.unpooling_2d(h, ksize=2, outsize=(96, 128))
        # print('after dconv1 unpooling output shape: ', h.data.shape)
        h = F.tanh(self.dconv1(h))
        # print('after dconv1 output shape: ', h.data.shape)
        # self.disp_img_arr(h)

        return h

    def encoder_net4(self, x):
        h = self.conv1(x)
        # print('after conv1 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.conv2(h)
        # print('after conv2 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.conv3(h)
        # print('after conv3 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.conv4(h)
        # print('after conv4 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.fc5(h)
        # print('after fc5 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
        h = F.dropout(h)

        h = self.fc6(h)
        # print('after fc6 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
        h = F.dropout(h)

        return h

    def decoder_net4(self, h, input_shape):
        h = self.dfc6(h)
        # print('after dfc6 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
        h = F.dropout(h)

        h = self.dfc5(h)
        # print('after dfc5 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
        h = F.dropout(h)

        h = self.dconv4(h.reshape(input_shape[0], 8, 24, 32))
        # print('after dconv4 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.dconv3(h)
        # print('after dconv3 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.dconv2(h)
        # print('after dconv2 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.dconv1(h)
        # print('after dconv1 output shape: ', h.data.shape)
        # h = F.leaky_relu(h)
        # h = F.tan(h)
        return h
    
    def encoder_net5(self, x):
        h = self.conv1(x)
#         print('after conv1 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.conv2(h)
#         print('after conv2 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.fc5(h)
#         print('after fc5 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
#         h = F.dropout(h, ratio=0.5)

        h = self.fc6(h)
        h = F.leaky_relu(h)
#         print('after fc6 output shape: ', h.data.shape)
#         h = F.leaky_relu(h)
#         h = F.dropout(h)

        return h
    
    def decoder_net5(self, h, input_shape):
        h = self.dfc6(h)
#         print('after dfc6 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
#         h = F.dropout(h)

        h = self.dfc5(h)
#         print('after dfc5 output shape: ', h.data.shape)
        h = F.leaky_relu(h)
#         h = F.dropout(h, ratio=0.5)

        h = self.dconv2(h.reshape(input_shape[0], 16, 96, 128))
#         h = self.dconv2(h)
#         print('after dconv2 output shape: ', h.data.shape)
        h = F.leaky_relu(h)

        h = self.dconv1(h)
#         print('after dconv1 output shape: ', h.data.shape)
        # h = F.leaky_relu(h)
        # h = F.tan(h)
        return h
    
    def encoder_net6(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h)

        h = self.conv2(h)
        h = F.leaky_relu(h)

        h = self.fc5(h)
        h = F.leaky_relu(h)
        # h = F.dropout(h, ratio=0.2)

        h = self.fc6(h)
        h = F.tanh(h)
        
        return h
    
    def decoder_net6(self, h, input_shape):
        h = self.dfc6(h)
        h = F.leaky_relu(h)
        # h = F.tanh(h)

        h = self.dfc5(h)
        h = F.leaky_relu(h)
        # h = F.dropout(h, ratio=0.2)

        h = self.dconv2(h.reshape(input_shape[0], 16, 96, 128))
        h = F.leaky_relu(h)

        h = self.dconv1(h)

        return h

    def encoder_layers(self, x):
        # self.disp_img_arr(x)
        if self.net_idx == 1:
            return self.encoder_net1(x)
        elif self.net_idx == 2:
            return self.encoder_net2(x)
        elif self.net_idx == 3:
            return self.encoder_net3(x)
        elif self.net_idx == 4:
            return self.encoder_net4(x)
        elif self.net_idx == 5:
            return self.encoder_net5(x)
        elif self.net_idx == 6:
            return self.encoder_net6(x)

    def decoder_layers(self, h, input_shape):
        if self.net_idx == 1:
            return self.decoder_net1(h, input_shape)
        elif self.net_idx == 2:
            return self.decoder_net2(h, input_shape)
        elif self.net_idx == 3:
            return self.decoder_net3(h, input_shape)
        elif self.net_idx == 4:
            return self.decoder_net4(h, input_shape)
        elif self.net_idx == 5:
            return self.decoder_net5(h, input_shape)
        elif self.net_idx == 6:
            return self.decoder_net6(h, input_shape)

    def __call__(self, x):
        input_shape = x.shape
        h_e = self.encoder_layers(x)
        h = self.decoder_layers(h_e, input_shape)
        return h

    def disp_img_arr(self, x):
        # print('x shape: ', x[0][0].shape)
        x_t = np.rollaxis(x[0], 0, 3)
        # print('x_t shape: ', x_t.shape)
        for h_idx in range(0, x_t.shape[0]):
            for w_idx in range(0, x_t.shape[1]):
                print('p({},{}):{}'.format(h_idx, w_idx, x_t[h_idx][w_idx][:]))


class ImageAE(object):
    """
    AutoEncoder for images.

    author: yajue
    date: 20171006
    """
    def __init__(self, network, max_epoch, batch_size, save_model_name):
        # network
        self.model = network

        # Optimizer
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)

        # training parameters
        self.max_epoch = max_epoch
        self.batch_size = batch_size

        self.save_model_name = save_model_name

    def training(self, train_data, test_data):
        model_root_name = r'/home/young/URLearning/model_gpu'
        # save train mean
        np.save(os.path.join(model_root_name, self.save_model_name.rstrip('.model') + 'meanImg.npy'), train_data.img_mean)
        
        # dataset iterator
        train_iter = iterators.SerialIterator(train_data, self.batch_size)
        test_iter = iterators.SerialIterator(test_data, self.batch_size)

#         loss_list = []
        loss_sum = cuda.cupy.zeros(1)
        in_batch_iter = 0
        train_start = time.time()
        epoch_start = time.time()

        while train_iter.epoch < self.max_epoch:
            batch_start = time.time()
            train_batch = train_iter.next()

            # model forward calculation
            train_batch_cupy = cuda.to_gpu(np.array(np.array(train_batch) - np.expand_dims(train_data.img_mean.transpose(2, 0, 1), 0), dtype=train_data._dtype))
#             train_batch_cupy = cuda.to_gpu(np.array(train_batch, dtype=train_data._dtype))
            pred = self.model(train_batch_cupy)
#             pred = self.model(np.array(train_batch))

            # loss
#             loss = F.mean_squared_error(pred, np.array(train_batch))
            loss = F.mean_squared_error(pred, train_batch_cupy)
#             loss_list.append(loss.data)
            loss_sum += loss.data

            # calculate the gradients in the networks
            self.model.cleargrads()

            loss.backward()

            self.optimizer.update()

#             print('batch_loss:{:.04f}, batch_elapsed_time:{:.04f} seconds'.format(float(loss.data), time.time()-batch_start))
            in_batch_iter += 1

            # check the validation accuracy of prediction after every epoch
            if train_iter.is_new_epoch:
                with chainer.using_config('train', False):
#                     test_losses = []
                    test_loss_sum = cuda.cupy.zeros(1)
                    test_batch_iter = 0
                    while True:
                        test_batch = test_iter.next()

                        # forward the test data
#                         test_batch_cupy = cuda.to_gpu(np.array(test_batch))
                        test_batch_cupy = cuda.to_gpu(np.array(np.array(test_batch) - np.expand_dims(train_data.img_mean.transpose(2, 0, 1), 0), dtype=train_data._dtype))
#                         pred_test = self.model(np.array(test_batch))
                        pred_test = self.model(test_batch_cupy)

                        # loss
#                         loss_test = F.mean_squared_error(pred_test, np.array(test_batch))
                        loss_test = F.mean_squared_error(pred_test, test_batch_cupy)
#                         test_losses.append(loss_test.data)
                        test_loss_sum += loss_test.data
                        test_batch_iter += 1

                        if test_iter.is_new_epoch:
                            test_iter.epoch = 0
                            test_iter.current_position = 0
                            test_iter.is_new_epoch = False
                            test_iter._pushed_position = None
                            break

                    # display
#                     train_loss_mean = np.mean(loss_list)
                    train_loss_mean = loss_sum / in_batch_iter
                    validate_loss_mean = test_loss_sum / test_batch_iter
#                     validate_loss_mean = np.mean(test_losses)
                    print('epoch:{:02d} train_loss:{:.04f} val_loss:{:.04f} epoch_elapsed_time:{:.04f} min'.format(train_iter.epoch, float(train_loss_mean), float(validate_loss_mean), (time.time() - epoch_start) / 60))

                    # save model
                    os.chdir(model_root_name)
                    serializers.save_npz(self.save_model_name, self.model)

                    # loss file and validation file
                    loss_file = ('TrainLoss' + self.save_model_name).rstrip('.model') + '.txt'
                    eval_file = ('EvalLoss' + self.save_model_name).rstrip('.model') + '.txt'

                    with open(os.path.join(model_root_name, loss_file), 'a') as f:
                        f.write(str(float(train_loss_mean)) + '\n')
                    with open(os.path.join(model_root_name, eval_file), 'a') as f:
                        f.write(str(float(validate_loss_mean)) + '\n')

                    epoch_start = time.time()
                    loss_sum = cuda.cupy.zeros(1)
                    in_batch_iter = 0

        print('training elapsed_time:{:.04f} hours'.format((time.time()-train_start) / 3600))


if __name__ == "__main__":
    """
    Train, validation, test dataset
    """
    train_imgs= []
    validate_imgs = []
    root_name = r'/home/young/URLearning'
    
    # train dataset
    for d_idx in range(1, 129):
        path_name = 'scoop_dataset6/group' + str(d_idx)
        os.chdir(os.path.join(root_name, path_name))
        images = glob.glob('*.jpg')
        for img_idx in range(len(images)):
            images[img_idx] = os.path.join(path_name, images[img_idx])
        if d_idx % 16 == 0:
            print('validate path: ', path_name)
            validate_imgs = validate_imgs + images
        else:
            print('training path: ', path_name)
            train_imgs = train_imgs + images

    print('train length: ', len(train_imgs))

    test_root_name = r'/home/young/URLearning/scoop_dataset6'

    test_imgs = []
    test_path_name = 'group112'
    os.chdir(os.path.join(test_root_name, test_path_name))
    test_imgs = glob.glob('*.jpg')
    for img_idx in range(len(test_imgs)):
        test_imgs[img_idx] = os.path.join(test_path_name, test_imgs[img_idx])

    train = TaskImageDataset(paths=train_imgs, root=root_name)
    test = TaskImageDataset(paths=test_imgs, root=test_root_name)
    validation = TaskImageDataset(paths=validate_imgs, root=root_name)

    # set up gpu
    gpu_device = 0
    chainer.cuda.get_device_from_id(gpu_device).use()
    model = ImageAENetwork(6)
    model.to_gpu()

    model_name = r'GPU_ImgAE45.model'

    # auto_encoder = ImageAE(model, 50, 16, model_name)
    # auto_encoder.training(train, validation)

    # load model
    model_root_name = r'/home/young/URLearning/model_gpu'
    serializers.load_npz(os.path.join(model_root_name, model_name), model)
    result_root_name = r'/home/young/URLearning/results/result52'

    # test
    test_iter = iterators.SerialIterator(test, 1, shuffle=False)
    train_test_iter = iterators.SerialIterator(train, 1)
    mean_img = np.load(os.path.join(model_root_name, model_name.rstrip('.model') + 'meanImg.npy'))
    mean_fig = Image.fromarray(np.array((mean_img[:, :, 0]+1) * 128, dtype=np.uint8), mode='L')

    # mean_fig.show()

    call_time = 0
    num_test = len(test_imgs)
    loss_sum = cuda.cupy.zeros(1)
    idx_loss = []
    with chainer.using_config('train', False):
        for i in range(0, num_test):
            test_batch = test_iter.next()
            pred_start = time.time()
#             test_batch_cupy = Variable(cuda.to_gpu(np.array(test_batch)))
            test_batch_cupy = cuda.to_gpu(np.array(np.array(test_batch) - np.expand_dims(mean_img.transpose(2, 0, 1), 0), dtype=train._dtype))
    #         pred = model.encoder_layers(test_batch_cupy)
            pred = model(test_batch_cupy)
    #         pred = model.encoder_layers(np.array(test_batch))
    #         pred = model(np.array(test_batch))
            call_time += time.time() - pred_start
            loss = F.mean_squared_error(pred, test_batch_cupy).data
            loss_sum += loss
    #         loss.append(F.mean_squared_error(pred, np.array(test_batch)).data)
            test_batch_name = str(i) + 'true.jpg'
#             test_img = Image.fromarray(np.array((np.rollaxis(test_batch[0], 0, 3) + 1) * 128, dtype=np.uint8))
            # grayscale
            test_img = Image.fromarray(np.array((np.rollaxis(test_batch[0], 0, 3)[:, :, 0] + 1) * 128, dtype=np.uint8), mode='L')
            test_img.save(os.path.join(result_root_name, test_batch_name))
            pred_name = str(i) + 'pred.jpg'
    #         pred_img = Image.fromarray(np.array((np.rollaxis(cuda.to_cpu(pred.data[0]), 0, 3) + 1) * 128, dtype=np.uint8))
#             pred_img = Image.fromarray(np.array((np.rollaxis(cuda.to_cpu(pred.data[0]), 0, 3) + mean_img + 1) * 128, dtype=np.uint8))
            # grayscale
            pred_img = Image.fromarray(np.array((np.rollaxis(cuda.to_cpu(pred.data[0]), 0, 3)[:, :, 0] + mean_img[:, :, 0] + 1) * 128, dtype=np.uint8), mode='L')
#             pred_img = Image.fromarray(np.array((np.rollaxis(cuda.to_cpu(pred.data[0]), 0, 3)[:, :, 0]+ 1) * 128, dtype=np.uint8), mode='L')
            pred_img.save(os.path.join(result_root_name, pred_name))
            idx_loss.append((i, float(loss)))

        print('avg call time: ', call_time / num_test)
        idx_loss.sort(key=lambda x: x[1])
        with open(os.path.join(result_root_name, 'loss_file.txt'), 'a') as f:
            for elem in idx_loss:
                f.write('[' + str(elem[0]) + ']' + '    ' + str(float(elem[1])) + '\n')
            f.write('average loss:' + str(float(loss_sum / num_test)) + '\n')
        print(float(loss_sum / num_test))
