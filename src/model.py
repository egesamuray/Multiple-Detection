from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import h5py
from module import *
from utils import *
from math import floor
from random import shuffle
from subprocess import Popen
import ntpath
import random

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size0 = args.image_size0
        self.image_size1 = args.image_size1
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.train_size = args.train_size
        self.transfer = args.transfer

        self.file_name_trainA = os.path.join(args.data_path, 'Nelson_A_recon2prim_reflect_train.hdf5')
        self.file_name_trainB = os.path.join(args.data_path, 'Nelson_B_recon2prim_reflect_train.hdf5')
        self.file_name_testA  = os.path.join(args.data_path, 'Nelson_A_recon2prim_reflect_test.hdf5')
        self.file_name_testB  = os.path.join(args.data_path, 'Nelson_B_recon2prim_reflect_test.hdf5')

        if self.transfer==1:
            self.file_name_trainA = self.file_name_testA
            self.file_name_trainB = self.file_name_testB

        self.test_case = args.test_case

        self.var1 = 'Nelson_A_recon-srmemult2prim_reflect_train_Fnet_ng32_nd64_poorPrediction_divide'

        self.dataset_name_train  = "train_dataset"
        self.dataset_name_test   = "test_dataset"

        if self.transfer==1:
            self.dataset_name_train = self.dataset_name_test

        self.file_trainA = h5py.File(self.file_name_trainA, 'r')
        self.file_trainB = h5py.File(self.file_name_trainB, 'r')
        self.file_testA  = h5py.File(self.file_name_testA, 'r')
        self.file_testB  = h5py.File(self.file_name_testB, 'r')

        self.data_num  = self.file_trainA[self.dataset_name_train].shape[0]
        self.test_num  = self.file_testB[self.dataset_name_test].shape[0]

        self.discriminator = discriminator
        if args.use_resnet==1:
            self.generator = generator_resnet
        elif args.use_resnet==2:
            self.generator = generator_FuisonNet            
        else:
            self.generator = generator_unet
        if args.use_lsgan==1:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size0 image_size1 \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.image_size0, args.image_size1,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=45)
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size0, self.image_size1, 4],
                                        name='real_A_and_B_images')

        self.equality_factor  = tf.placeholder(tf.float32, None, name='equality_factor')
        self.SNR_diff = tf.placeholder(tf.float32, [None, self.image_size0*self.image_size1], name='SNR_diff')
        self.SNR_real = tf.placeholder(tf.float32, [None, self.image_size0*self.image_size1], name='SNR_real')

        self.real_A = self.real_data[:, :, :, 1:1 + self.output_c_dim]
        self.real_B = self.real_data[:, :, :, 2:2 + self.input_c_dim]
        print(self.real_data.shape)

        self.fake_A = self.generator(self.real_B, self.options, False, name="generatorB2A")

        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))

        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.equality_factor * abs_criterion(self.fake_A, self.real_A)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size0, self.image_size1,
                                             self.output_c_dim], name='fake_A_sample')

        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss

        self.Rec_SNR = -20.0* tf.log(tf.norm(self.SNR_diff, ord='euclidean')/tf.norm(self.SNR_real, ord='euclidean'))/tf.log(10.0)

        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_b2a_sum, self.g_loss_sum])
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.d_loss_sum]
        )
        self.Rec_SNR_train_sum = tf.summary.scalar("Rec_SNR_train", self.Rec_SNR)
        self.Rec_SNR_test_sum  = tf.summary.scalar("Rec_SNR_test", self.Rec_SNR)

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size0, self.image_size1,
                                      self.output_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size0, self.image_size1,
                                      self.input_c_dim], name='test_B')
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()

        if self.transfer==0:
            self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
        elif self.transfer==1:
            self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            self.d_vars  = self.d_vars[-4:]
            self.g_vars  = self.g_vars[9:]

        for var in self.d_vars: print(var.name)
        for var in self.g_vars: print(var.name)        

        var_size = 0
        for var in self.d_vars:
            var_size = var_size + int(np.prod(np.array(var.shape)))
        for var in self.g_vars:
            var_size = var_size + int(np.prod(np.array(var.shape)))
        print(("Number of unknowns: %d" % (var_size)))

    def train(self, args):
        """Train cyclegan"""

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if self.transfer==0:
            batch_idxs = list(range(min(int(floor(float(self.data_num) / self.batch_size)), self.train_size)))
        elif self.transfer==1:
            batch_idxs = list(range(0, min(int(floor(float(self.data_num) / self.batch_size)), self.train_size), 5))

        for epoch in range(args.epoch):

            shuffle(batch_idxs)
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            equality_factor = 0.0*(epoch) + self.L1_lambda

            for idx in range(0, len(batch_idxs)):

                batch_images = load_train_data(batch_idxs[idx], batch_size=self.batch_size, \
                    fileA=self.file_trainA, fileB=self.file_trainB, dataset=self.dataset_name_train)
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_A, _, summary_str = self.sess.run(
                    [self.fake_A, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr, self.equality_factor: equality_factor})
                self.writer.add_summary(summary_str, counter)

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, int(idx), int(len(batch_idxs)), time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, counter-1)

                if np.mod(counter, int(floor(args.save_freq/self.batch_size))) == 2:
                    self.save(args.checkpoint_dir, counter)
                    Process=Popen('/home/ec2-user/AWSscripts/UploadToDropbox.sh %s' % (str(self.var1)), shell=True)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx, counter):

        if self.transfer==0:
            res_rnd = int(np.random.randint(0, self.data_num))
        elif self.transfer==1:
                    res_rnd = int(random.choice(list(range(0, \
                        min(int(floor(float(self.data_num) / self.batch_size)), self.train_size), 5))))

        train_images = load_train_data(res_rnd, is_testing=True, batch_size=self.batch_size, \
            fileA=self.file_trainA, fileB=self.file_trainB, dataset=self.dataset_name_train)
        train_images = np.array(train_images).astype(np.float32)

        sample_A = train_images[:, :, :, 1:1 + self.output_c_dim]
        sample_B = train_images[:, :, :, 2:2 + self.input_c_dim]

        out_var, in_var = (self.testA, self.test_B)

        fake_A = self.sess.run(out_var, feed_dict={in_var: sample_B})

        result_img = fake_A
        diff_img = np.absolute(sample_A-result_img)

        diff_img_real = diff_img[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))
        sample_A_real = sample_A[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))

        Rec_SNR, summary_str = self.sess.run(
            [self.Rec_SNR, self.Rec_SNR_train_sum],
            feed_dict={self.SNR_diff: diff_img_real,
                       self.SNR_real: sample_A_real})
        self.writer.add_summary(summary_str, counter)

        print(("Recovery SNR (training data): %4.4f" % (Rec_SNR)))

#################################

        res_rnd = int(np.random.randint(0, self.test_num))

        sample_B = load_test_data(res_rnd, filetest=self.file_testB, dataset=self.dataset_name_test)
        sample_B = np.array(sample_B).astype(np.float32)[:, :, :, :self.input_c_dim]

        sample_A = load_test_data(res_rnd, filetest=self.file_testA, dataset=self.dataset_name_test)
        sample_A = np.array(sample_A).astype(np.float32)[:, :, :, 1:1+self.output_c_dim]

        fake_A = self.sess.run(out_var, feed_dict={in_var: sample_B})

        result_img = fake_A
        diff_img = np.absolute(sample_A-result_img)

        diff_img_real = diff_img[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))
        sample_A_real = sample_A[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))

        Rec_SNR, summary_str = self.sess.run(
            [self.Rec_SNR, self.Rec_SNR_test_sum],
            feed_dict={self.SNR_diff: diff_img_real,
                       self.SNR_real: sample_A_real})
        self.writer.add_summary(summary_str, counter)

        print(("Recovery SNR (testing data): %4.4f" % (Rec_SNR)))


    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        out_var, in_var = (self.testA, self.test_B)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

####################################

        res_rnd = int(np.random.randint(0, self.test_num))#self.test_case

####################################

        sample_B = load_test_data(res_rnd, filetest=self.file_testB, dataset=self.dataset_name_test)
        sample_B = np.array(sample_B).astype(np.float32)[:, :, :, :self.input_c_dim]

        sample_file = 'gradient'

        print('Processing test image: ' + str(res_rnd))
        start_time_interp = time.time()
        fake_img = self.sess.run(out_var, feed_dict={in_var: sample_B})
        print(("Mapping time: %4.4f seconds" % (time.time() - start_time_interp)))

        sample_A = load_test_data(res_rnd, filetest=self.file_testA, dataset=self.dataset_name_test)
        sample_A = np.array(sample_A).astype(np.float32)[:, :, :, 1:1+self.output_c_dim]

        fake_output={}
        fake_output['result'] = fake_img
        fake_output['original'] = sample_A
        fake_output['masked'] = sample_B

        io.savemat('mapping_result{0}_{1}'.format(args.which_direction, os.path.basename(str(sample_file))),fake_output)

        result_img = fake_img
        diff_img = np.absolute(sample_A-result_img)

        diff_img_real = diff_img[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))
        sample_A_real = sample_A[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))

        Rec_SNR_real = self.sess.run(
            [self.Rec_SNR],
            feed_dict={self.SNR_diff: diff_img_real,
                       self.SNR_real: sample_A_real})

        print(("Recovery SNR (test dataset): %4.4f" % (Rec_SNR_real[0])))

        SNR_AVG0 = 0
        SNR_AVG1 = 0
        iii = 0

        strResult = os.path.join(self.sample_dir, 'mapping_SNR.hdf5')

        if os.path.isfile(strResult):
            os.remove(strResult)

        file_SNR = h5py.File(strResult, 'w-')
        dataset_str = "SNR"
        datasetSNR = file_SNR.create_dataset(dataset_str, (1, 1))

        strCorrection = os.path.join(self.sample_dir, 'mapping_result.hdf5')

        if os.path.isfile(strCorrection):
            os.remove(strCorrection)

        file_correction = h5py.File(strCorrection, 'w-')
        datasetCorrection_str = "result"
        datasetCorrection = file_correction.create_dataset(datasetCorrection_str, \
            (self.test_num, self.image_size0, self.image_size1, self.output_c_dim))        
        datasetCorrectionA = file_correction.create_dataset(datasetCorrection_str + \
            'A', (self.test_num, self.image_size0, self.image_size1, 2))
        datasetCorrectionB = file_correction.create_dataset(datasetCorrection_str + \
            'B', (self.test_num, self.image_size0, self.image_size1, 2))

        start_time_interp = time.time()

        for itest in range(0, self.test_num):
            print(itest)
            res_rnd = itest
            # res_rnd =
            sample_B = load_test_data(res_rnd, filetest=self.file_testB, dataset=self.dataset_name_test)
            sample_B = np.array(sample_B).astype(np.float32)

            sample_A = load_test_data(res_rnd, filetest=self.file_testA, dataset=self.dataset_name_test)
            sample_A = np.array(sample_A).astype(np.float32)[:, :, :, 1:1+self.output_c_dim]

            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_B[:, :, :, :self.input_c_dim]})
            datasetCorrection[itest, :, :, :self.output_c_dim] = fake_img[0, :, :, :self.output_c_dim]
            datasetCorrectionB[itest, :, :, :] = sample_B[0, :, :, :]
            datasetCorrectionA[itest, :, :, :] = sample_A[0, :, :, :]

            result_img = fake_img
            diff_img = np.absolute(sample_A-result_img)

            diff_img_real = diff_img[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))
            sample_A_real = sample_A[:, :, :, 0].reshape((1, self.image_size0*self.image_size1))

            Rec_SNR_real = self.sess.run(
                [self.Rec_SNR],
                feed_dict={self.SNR_diff: diff_img_real,
                           self.SNR_real: sample_A_real})

            print(("Recovery SNR for shot: %4.4f" % (Rec_SNR_real[0])))

            SNR_AVG0 = SNR_AVG0 + Rec_SNR_real[0]
            # from IPython import embed; embed()
            iii = iii + 1

        datasetSNR[0, 0] = SNR_AVG0/iii
        print(datasetSNR[0, 0])
        file_SNR.close()
        file_correction.close()