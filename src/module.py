from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import numpy as np

def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        print(np.shape(image))

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        print(np.shape(h0))

        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, ks=[4, 4], name='d_h1_conv'), 'd_bn1'))
        print(np.shape(h1))

        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, ks=[4, 4], name='d_h2_conv'), 'd_bn2'))
        print(np.shape(h2))

        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=[4, 4], s=1, name='d_h3_conv'), 'd_bn3'))
        print(np.shape(h3))

        h4 = conv2d(h3, 1, ks=[4, 4], s=1, name='d_h3_pred')
        print(np.shape(h4))

        return h4


def generator_unet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            # p = int((ks - 1) / 2)
            p = [int((x-1)/2) for x in ks]
            y = tf.pad(x, [[0, 0], [p[0], p[0]+1], [p[1], p[1]], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p[0], p[0]+1], [p[1], p[1]], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return tf.nn.relu(y + x)

        print(np.shape(image))
        c0 = tf.pad(image, [[0, 0], [8, 7], [3, 3], [0, 0]], "REFLECT")
        print(np.shape(c0))
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, [14, 7], 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        print(np.shape(c1))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, [6, 3], 2, name='g_e2_c'), 'g_e2_bn'))
        print(np.shape(c2))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, [6, 3], 2, name='g_e3_c'), 'g_e3_bn'))
        print(np.shape(c3))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, ks=[6, 3], name='g_r1')
        print(np.shape(r1))
        r2 = residule_block(r1, options.gf_dim*4, ks=[6, 3], name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, ks=[6, 3], name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, ks=[6, 3], name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, ks=[6, 3], name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, ks=[6, 3], name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, ks=[6, 3], name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, ks=[6, 3], name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, ks=[6, 3], name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, [6, 3], 2, name='g_d1_dc')
        print(np.shape(d1))
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, [6, 3], 2, name='g_d2_dc')
        print(np.shape(d2))
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [6, 5], [3, 3], [0, 0]], "REFLECT")
        print(np.shape(d2))
        pred = (conv2d(d2, options.output_c_dim, [14, 7], 1, padding='VALID', name='g_pred_c'))
        print(np.shape(pred))

        return pred



def generator_FuisonNet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=4, s=1, name='res'):

            y = instance_norm(conv2d(x, dim, ks, s, name=name+'_c1'), name+'_bn1')
            y = instance_norm(conv2d(lrelu(y), dim, ks, s, name=name+'_c2'), name+'_bn2')
            return y + x

        print('image')
        print(image.shape)
        image = tf.pad(image, [[0, 0], [0, 0], [55, 56], [0, 0]], "REFLECT")
        # image is (256 x 256 x input_c_dim)
        r1 = lrelu(residule_block(image, image.shape[3], name='g_e_r1'))
        e1 = instance_norm(conv2d(r1, options.gf_dim, name='g_e1_conv'))
        print('e1')
        print(e1.shape)
        # e1 is (128 x 128 x self.gf_dim)
        r2 = lrelu(residule_block(lrelu(e1), options.gf_dim, name='g_e_r2'))
        e2 = instance_norm(conv2d(r2, options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        print('e2')
        print(e2.shape)
        # e2 is (64 x 64 x self.gf_dim*2)
        r3 = lrelu(residule_block(lrelu(e2), options.gf_dim*2, name='g_e_r3'))
        e3 = instance_norm(conv2d(r3, options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        print('e3')
        print(e3.shape)
        # e3 is (32 x 32 x self.gf_dim*4)
        r4 = lrelu(residule_block(lrelu(e3), options.gf_dim*4, name='g_e_r4'))
        e4 = instance_norm(conv2d(r4, options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        print('e4')
        print(e4.shape)
        # e4 is (16 x 16 x self.gf_dim*8)
        r5 = lrelu(residule_block(lrelu(e4), options.gf_dim*8, name='g_e_r5'))
        e5 = instance_norm(conv2d(r5, options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        print('e5')
        print(e5.shape)
        # e5 is (8 x 8 x self.gf_dim*8)
        r6 = lrelu(residule_block(lrelu(e5), options.gf_dim*8, name='g_e_r6'))
        e6 = instance_norm(conv2d(r6, options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        print('e6')
        print(e6.shape)
        # e6 is (4 x 4 x self.gf_dim*8)
        r7 = lrelu(residule_block(lrelu(e6), options.gf_dim*8, name='g_e_r7'))
        e7 = instance_norm(conv2d(r7, options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        print('e7')
        print(e7.shape)
        # e7 is (2 x 2 x self.gf_dim*8)
        r8 = lrelu(residule_block(lrelu(e7), options.gf_dim*8, name='g_e_r8'))
        e8 = instance_norm(conv2d(r8, options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        print('e8')
        print(e8.shape)
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        d1 = residule_block(lrelu(d1), options.gf_dim*16, name='g_e_r9')
        print('d1')
        print(d1.shape)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        d2 = residule_block(lrelu(d2), options.gf_dim*16, name='g_e_r10')
        print('d2')
        print(d2.shape)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        d3 = residule_block(lrelu(d3), options.gf_dim*16, name='g_e_r11')
        print('d3')
        print(d3.shape)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        d4 = residule_block(lrelu(d4), options.gf_dim*16, name='g_e_r12')
        print('d4')
        print(d4.shape)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        d5 = residule_block(lrelu(d5), options.gf_dim*8, name='g_e_r13')
        print('d5')
        print(d5.shape)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        d6 = residule_block(lrelu(d6), options.gf_dim*4, name='g_e_r14')
        print('d6')
        print(d6.shape)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        d7 = residule_block(lrelu(d7), options.gf_dim*2, name='g_e_r15')
        print('d7')
        print(d7.shape)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        d8 = residule_block(lrelu(d8), options.output_c_dim, name='g_e_r16')
        print('d8')
        print(d8[:, :, 55:512-56, :].shape)
        # d8 is (256 x 256 x output_c_dim)

        return d8[:, :, 55:512-56, :]


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
