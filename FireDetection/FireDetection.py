# -*- coding: utf-8 -*-
from tensorflow.python.keras import backend as K
from functools import partial
import tensorflow as tf
import warnings


class FireDetection(object):
    def __init__(self):
        pass

    def preprocess_input(self, x):
        """
        Preprocesses a numpy array encoding a batch of images.
        # Arguments
            x: a 4D numpy array consists of RGB values within [0, 255].
        # Returns
            Preprocessed array.
        """
        return ((x / 255.) - 0.5) * 2.

    def generate_layer_name(self, name, branch_idx=None, prefix=None):
        """
        Utility function for generating layer names.
        # Arguments
            name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
            branch_idx: an `int`. If given, will add e.g. `'Branch_0'` after `prefix` and in front of `name` in order to
                identify layers in the same block but in different branches.
            prefix: string prefix that will be added in front of `name` to make all layer names unique (e.g. which block
                this layer belongs to).
        # Returns
            The layer name.
        """
        if prefix is None:
            return None

        if branch_idx is None:
            return '_'.join((prefix, name))

        return '_'.join((prefix, 'Branch', str(branch_idx), name))

    def conv2d_bn(self, x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
        """
        Utility function to apply conv + BN.
        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            kernel_size: kernel size as in `Conv2D`.
            padding: padding mode in `Conv2D`.
            activation: activation in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_Activation'` for the activation and `name + '_BatchNorm'` for
                the batch norm layer.
        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
        if not use_bias:
            bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
            bn_name = self.generate_layer_name('BatchNorm', prefix=name)
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        if activation is not None:
            ac_name = self.generate_layer_name('Activation', prefix=name)
            x = tf.keras.layers.Activation(activation, name=ac_name)(x)

        return x

    def inception_resnet_block(self, x, scale, block_type, block_idx, activation='relu'):
        """
        Adds a Inception-ResNet block.
        # Arguments
            x: input tensor.
            scale: scaling factor to scale the residuals before adding them to the shortcut branch.
            block_type: `'Block35'`, `'Block17'` or `'Block8'`, determines the network structure in the residual branch.
            block_idx: used for generating layer names.
            activation: name of the activation function to use at the end of the block(see [activations](../activations.md)).
                When `activation=None`, no activation is applied(i.e., "linear" activation: `a(x) = x`).
        # Returns
            Output tensor for the block.
        # Raises
            ValueError: if `block_type` is not one of `'Block35'`, `'Block17'` or `'Block8'`.
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        if block_idx is None:
            prefix = None
        else:
            prefix = '_'.join((block_type, str(block_idx)))

        name_fmt = partial(self.generate_layer_name, prefix=prefix)

        if block_type == 'Block35':
            branch_0 = self.conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
            branch_1 = self.conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
            branch_1 = self.conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
            branch_2 = self.conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
            branch_2 = self.conv2d_bn(branch_2, 48, 3, name=name_fmt('Conv2d_0b_3x3', 2))
            branch_2 = self.conv2d_bn(branch_2, 64, 3, name=name_fmt('Conv2d_0c_3x3', 2))
            branches = [branch_0, branch_1, branch_2]
        elif block_type == 'Block17':
            branch_0 = self.conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
            branch_1 = self.conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
            branch_1 = self.conv2d_bn(branch_1, 160, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
            branch_1 = self.conv2d_bn(branch_1, 192, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
            branches = [branch_0, branch_1]
        elif block_type == 'Block8':
            branch_0 = self.conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
            branch_1 = self.conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
            branch_1 = self.conv2d_bn(branch_1, 224, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
            branch_1 = self.conv2d_bn(branch_1, 256, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "Block35", "Block17" or "Block8", '
                             'but got: ' + str(block_type))

        mixed = tf.keras.layers.Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)
        up = self.conv2d_bn(mixed, K.int_shape(x)[channel_axis], 1,
                            activation=None,
                            use_bias=True,
                            name=name_fmt('Conv2d_1x1'))
        x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                                   output_shape=K.int_shape(x)[1:],
                                   arguments={'scale': scale},
                                   name=name_fmt('ScaleSum'))([x, up])
        if activation is not None:
            x = tf.keras.layers.Activation(activation, name=name_fmt('Activation'))(x)

        return x

    def inception_resnet_v2(self, include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                            classes=1000, dropout_keep_prob=0.8):
        """
        Instantiates the Inception-ResNet v2 architecture.
        Optionally loads weights pre-trained on ImageNet.
        # Arguments
            include_top: whether to include the fully-connected layer at the top of the network.
            weights: one of `None` (random initialization) or `'imagenet'` (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the models.
            input_shape: optional shape tuple, only to be specified if `include_top` is `False` (otherwise the input
                shape has to be `(299, 299, 3)` (with `channels_last` data format) or `(3, 299, 299)`.
            pooling: Optional pooling mode for feature extraction when `include_top` is `False`.
                - `None` means that the output of the models will be the 4D tensor output of the last convolutional layer.
                - `'avg'` means that global average pooling will be applied to the output of thelast convolutional layer,
                    and thus the output of the models will be a 2D tensor.
                - `'max'` means that global max pooling will be applied.
            classes: optional number of classes to classify images into, only to be specified if `include_top` is `True`,
                and if no `weights` argument is specified.
            dropout_keep_prob: dropout keep rate after pooling and before the classification layer, only to be specified
                if `include_top` is `True`.
        # Returns
            A Keras `Model` instance.
        # Raises
            ValueError: in case of invalid argument for `weights`, or invalid input shape.
        """
        # Determine proper input shape
        if input_shape is None:
            input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = tf.keras.layers.Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        # Stem block: 35 x 35 x 192
        x = self.conv2d_bn(img_input, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
        x = self.conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
        x = self.conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        x = self.conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
        x = self.conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
        x = tf.keras.layers.MaxPooling2D(3, strides=2, name='MaxPool_5a_3x3')(x)

        # Mixed 5b (Inception-A block): 35 x 35 x 320
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        name_fmt = partial(self.generate_layer_name, prefix='Mixed_5b')
        branch_0 = self.conv2d_bn(x, 96, 1, name=name_fmt('Conv2d_1x1', 0))
        branch_1 = self.conv2d_bn(x, 48, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = self.conv2d_bn(branch_1, 64, 5, name=name_fmt('Conv2d_0b_5x5', 1))
        branch_2 = self.conv2d_bn(x, 64, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = self.conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = self.conv2d_bn(branch_2, 96, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branch_pool = tf.keras.layers.AveragePooling2D(3, strides=1, padding='same', name=name_fmt('AvgPool_0a_3x3', 3))(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, name=name_fmt('Conv2d_0b_1x1', 3))
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = tf.keras.layers.Concatenate(axis=channel_axis, name='Mixed_5b')(branches)

        # 10x Block35 (Inception-ResNet-A block): 35 x 35 x 320
        for block_idx in range(1, 11):
            x = self.inception_resnet_block(x, scale=0.17, block_type='Block35', block_idx=block_idx)

        # Mixed 6a (Reduction-A block): 17 x 17 x 1088
        name_fmt = partial(self.generate_layer_name, prefix='Mixed_6a')
        branch_0 = self.conv2d_bn(x, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))
        branch_1 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = self.conv2d_bn(branch_1, 256, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        branch_1 = self.conv2d_bn(branch_1, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))
        branch_pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid', name=name_fmt('MaxPool_1a_3x3', 2))(x)
        branches = [branch_0, branch_1, branch_pool]
        x = tf.keras.layers.Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

        # 20x Block17 (Inception-ResNet-B block): 17 x 17 x 1088
        for block_idx in range(1, 21):
            x = self.inception_resnet_block(x, scale=0.1, block_type='Block17', block_idx=block_idx)

        # Mixed 7a (Reduction-B block): 8 x 8 x 2080
        name_fmt = partial(self.generate_layer_name, prefix='Mixed_7a')
        branch_0 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
        branch_0 = self.conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 0))
        branch_1 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = self.conv2d_bn(branch_1, 288, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 1))
        branch_2 = self.conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = self.conv2d_bn(branch_2, 288, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = self.conv2d_bn(branch_2, 320, 3, strides=2, padding='valid', name=name_fmt('Conv2d_1a_3x3', 2))
        branch_pool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='valid', name=name_fmt('MaxPool_1a_3x3', 3))(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = tf.keras.layers.Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

        # 10x Block8 (Inception-ResNet-C block): 8 x 8 x 2080
        for block_idx in range(1, 10):
            x = self.inception_resnet_block(x, scale=0.2, block_type='Block8', block_idx=block_idx)
        x = self.inception_resnet_block(x, scale=1., activation=None, block_type='Block8', block_idx=10)

        # Final convolution block
        x = self.conv2d_bn(x, 1536, 1, name='Conv2d_7b_1x1')

        if include_top:
            # Classification block
            x = tf.keras.layers.GlobalAveragePooling2D(name='AvgPool')(x)
            x = tf.keras.layers.Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
            x = tf.keras.layers.Dense(classes, name='Logits')(x)
            x = tf.keras.layers.Activation('softmax', name='Predictions')(x)
        else:
            if pooling == 'avg':
                x = tf.keras.layers.GlobalAveragePooling2D(name='AvgPool')(x)
            elif pooling == 'max':
                x = tf.keras.layers.GlobalMaxPooling2D(name='MaxPool')(x)

        # Ensure that the models takes into account any potential predecessors of `input_tensor`
        if input_tensor is not None:
            inputs = tf.keras.utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # Create models
        model = tf.keras.models.Model(inputs, x, name='inception_resnet_v2')

        # Load weights - imagenet
        if weights == 'imagenet':
            BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'
            if K.image_data_format() == 'channels_first':
                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image data format convention '
                                  '(`image_data_format="channels_first"`). '
                                  'For best performance, set '
                                  '`image_data_format="channels_last"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
            if include_top:
                weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
                weights_path = tf.keras.utils.get_file(weights_filename,
                                        BASE_WEIGHT_URL + weights_filename,
                                        cache_subdir='models',
                                        md5_hash='e693bd0210a403b3192acc6073ad2e96')
            else:
                weights_filename = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
                weights_path = tf.keras.utils.get_file(weights_filename,
                                        BASE_WEIGHT_URL + weights_filename,
                                        cache_subdir='models',
                                        md5_hash='d19885ff4a710c122648d3b5c3b684e4')
            model.load_weights(weights_path)

        return model


if __name__ == '__main__':
    pass
