"""
Processing image-motion dataset and time-delay neural network for learning image-motion sequences.

author: yajue
date: 20171009
"""
import re, os, glob, random, time
from chainer import serializers, Chain, cuda, Variable
import chainer.links as L
import chainer.functions as F
import chainer.optimizers as O
from img_autoencoder_gpu import ImageAENetwork, TaskImageDataset
import numpy as np
import chainer
import utils
import sys
import matplotlib.pyplot as plt


class ImgMotLSTM(Chain):
    def __init__(self, net_idx, train=True):
        super(ImgMotLSTM, self).__init__()
        with self.init_scope():
            self.net_idx = net_idx
            
            if self.net_idx == 1:
                self.l1 = L.Linear(26, 150)
                self.l2 = L.LSTM(150, 150)
                self.l3 = L.Linear(150, 26)
                
            elif self.net_idx == 2:
                self.l1 = L.LSTM(36, 15)
                self.l2 = L.Linear(15, 6)
                
            elif self.net_idx == 3:
                self.l1 = L.Linear(26, 150)
                self.l2 = L.LSTM(150, 150)
                self.l3 = L.LSTM(150, 150)
                self.l4 = L.Linear(150, 26)
    
            elif self.net_idx == 4:
                self.l1 = L.Linear(16, 150)
                self.l2 = L.LSTM(150, 150)
                self.l3 = L.LSTM(150, 150)
                self.l4 = L.Linear(150, 16)

            elif self.net_idx == 5:
                self.l1 = L.Linear(16, 10)
                self.l2 = L.LSTM(10, 10)
                self.l3 = L.Linear(10, 16)

            else:
                print('ERROR: No network{}'.format(self.net_idx))
                sys.exit()
                    
            self.train = train
            
    def run_net1(self, x):
        h = self.l1(x)
        h = self.l2(h)
        y = self.l3(h)
        return y
    
    def run_net2(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y
    
    def run_net3(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        y = self.l4(h)
        return y
    
    def run_net4(self, x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        y = self.l4(h)
        return y

    def run_net5(self, x):
        h = self.l1(x)
        h = F.leaky_relu(h)
        # h = F.dropout(h, ratio=0.2)
        h = self.l2(h)
        h = F.leaky_relu(h)
        y = self.l3(h)
        return y
            
    def __call__(self, x, t):
        if self.net_idx == 1:
            y = self.run_net1(x)
        elif self.net_idx == 2:
            y = self.run_net2(x)
        elif self.net_idx == 3:
            y = self.run_net3(x)
        elif self.net_idx == 4:
            y = self.run_net4(x)
        elif self.net_idx == 5:
            y = self.run_net5(x)
        else:
            print('ERROR: No network{}'.format(self.net_idx))
            sys.exit()
        self.loss = F.mean_squared_error(y, t)
        if self.train:
            return self.loss
        else:
            self.prediction = y
            return self.prediction
    
    def reset_state(self):
        if self.net_idx == 1:
            self.l2.reset_state()
        elif self.net_idx == 2:
            self.l1.reset_state()
        elif self.net_idx == 3:
            self.l2.reset_state()
            self.l3.reset_state()
        elif self.net_idx == 4:
            self.l2.reset_state()
            self.l3.reset_state()
        elif self.net_idx == 5:
            self.l2.reset_state()
        else:
            print('ERROR: No network{}'.format(self.net_idx))
            sys.exit()


class IJSeqMaker(object):
    def __init__(self, img_model_file, img_mot_paths, imgnet_idx, time_window, img_feature_dim):
        """
        Initialization.
        :param img_model_file: Existing well trained Image AutoEncoder model file name
        :param img_mot_paths: A list of paths of directories containing images and robot motion files
        :param index of image autoencoder network
        :param time_window: Time window
        :param img_feature_dim: Dimension of image feature

        author: yajue
        date: 20171023
        """
        # load a well trained Image AutoEncoder network
        self.img_model = ImageAENetwork(imgnet_idx)
        print('image network index: ', imgnet_idx)
        serializers.load_npz(img_model_file, self.img_model)
        
        # load mean image
        mean_img_name = img_model_file.rstrip('.model') + 'meanImg.npy'
        self.mean_img = np.load(mean_img_name)
#         print('mean image shape: ', self.mean_img.shape)
        
        self.img_mot_paths = []
        self.img_mot_paths = img_mot_paths
        self.time_window = time_window
        self.img_feature_dim = img_feature_dim
        self.img_mot_dim = self.img_feature_dim + 6
        
        self.dataset = []
        
        self.num_data = len(self.dataset)
        
        # min-max joint angles
        self.joint_minmax = np.ndarray((2, 6))

        # average image feature and joint angles
        self.avg_imgft = None
        self.avg_jt = None
        
    def calc_joint_minmax(self, joint_angles):
        print(joint_angles.shape)
        jtmin_along_grp = np.min(joint_angles, axis=0)
        jtmin = np.min(jtmin_along_grp, axis=0)
        jtmax_along_grp = np.max(joint_angles, axis=0)
        jtmax = np.max(jtmax_along_grp, axis=0)
        self.joint_minmax[0, :] = jtmin - 0.02
        self.joint_minmax[1, :] = jtmax[:] + 0.02
        print('joint min max: ', self.joint_minmax)

    def numerical_sort(self, value):
        """
        Splits out any digits in a filename, turns it into an actual number, and returns the result for sorting.
        :param value: filename
        :return:

        author: yajue
        date: 20171009
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    def gen_dataset(self, start_steps, step_length):
        print('img mot paths: ', self.img_mot_paths)
        
        total_jt = np.empty((0, step_length, 6))
        
        for path_idx in range(len(self.img_mot_paths)):
            # get all images and extract features
            path = self.img_mot_paths[path_idx]
            os.chdir(path)
            img_features = []
            img_names = sorted(glob.glob('*.jpg'), key=self.numerical_sort)
            img_dataset = TaskImageDataset(img_names, path)
            
            for img_idx in range(start_steps[path_idx], start_steps[path_idx]+step_length):
                # grayscale
                img_arr = img_dataset.get_example(img_idx) - self.mean_img.transpose(2, 0, 1)
                with chainer.using_config('train', False):
                    feature = self.img_model.encoder_layers(np.expand_dims(img_arr, axis=0))
                assert feature.size == self.img_feature_dim
                # feature data shape: (1, feature_dim)
                img_features.append(feature.data.flatten())
                # print(feature.data.shape)
            
            # get all joint angles from the file
            joint_file = os.path.join(path, 'joint_position.txt')
            joint_angles = utils.load_joint_seq(joint_file, start_steps[path_idx], step_length)
            joint_angles = np.array(joint_angles)
            total_jt = np.append(total_jt, joint_angles[np.newaxis, :, :], axis=0)
                    
            # concatenate image features and joint angles
            assert len(img_features) == len(joint_angles)
            seq_length = len(img_features)
            for idx in range(0, seq_length - self.time_window + 1):
                img_joint = np.ndarray((self.img_mot_dim, self.time_window), dtype=np.float32)
                for seq_idx in range(0, self.time_window):
                    img_joint[0:self.img_feature_dim, seq_idx] = img_features[idx + seq_idx].transpose()
                    img_joint[self.img_feature_dim:self.img_feature_dim+6, seq_idx] = joint_angles[idx + seq_idx].transpose()
                self.dataset.append(img_joint)
        
        self.num_data = len(self.dataset)
        print('Number of sequences: ', len(self.dataset))
        
        self.calc_joint_minmax(total_jt)

    def gen_zc_dataset(self, start_steps, step_length):
        """
        Generate zero-centered dataset
        :param start_steps: starting step
        :param step_length: length of the whole trajectory
        :return:

        author: yajue
        date: 20171117
        """
        # load all dataset to calculate average

        total_imgft = np.empty((0, step_length, self.img_feature_dim))
        total_jt = np.empty((0, step_length, 6))

        for path_idx in range(len(self.img_mot_paths)):
            # images
            path = self.img_mot_paths[path_idx]
            print(path)
            os.chdir(path)
            img_names = sorted(glob.glob('*.jpg'), key=self.numerical_sort)
            img_dataset = TaskImageDataset(img_names, path)

            grp_imgft = np.empty((0, self.img_feature_dim))

            for img_idx in range(start_steps[path_idx], start_steps[path_idx]+step_length):
                img_arr = img_dataset.get_example(img_idx) - self.mean_img.transpose(2, 0, 1)
                with chainer.using_config('train', False):
                    feature = self.img_model.encoder_layers(np.expand_dims(img_arr, axis=0))
                assert feature.size == self.img_feature_dim
                grp_imgft = np.append(grp_imgft, feature.data, axis=0)

            total_imgft = np.append(total_imgft, grp_imgft[np.newaxis, :, :], axis=0)

            # joint angles
            joint_file = os.path.join(path, 'joint_position.txt')
            joint_angles = utils.load_joint_seq(joint_file, start_steps[path_idx], step_length)
            joint_angles = np.array(joint_angles)
            total_jt = np.append(total_jt, joint_angles[np.newaxis, :, :], axis=0)

        # average data
        self.avg_imgft = np.average(total_imgft, axis=0)
        self.avg_jt = np.average(total_jt, axis=0)

        # zero center data
        total_imgft = total_imgft - self.avg_imgft
        total_jt = total_jt - self.avg_jt

        # print('total imgft: ', total_imgft)
        # print('total jt: ', total_jt)

        # concatenate image features and joint angles
        for path_idx in range(len(self.img_mot_paths)):
            for idx in range(0, step_length - self.time_window + 1):
                img_joint = np.ndarray((self.img_mot_dim, self.time_window), dtype=np.float32)
                for seq_idx in range(0, self.time_window):
                    img_joint[0:self.img_feature_dim, seq_idx] = total_imgft[path_idx, idx+seq_idx, :].transpose()
                    img_joint[-6:, seq_idx] = total_jt[path_idx, idx+seq_idx, :].transpose()
                self.dataset.append(img_joint)

        self.num_data = len(self.dataset)
        print('Number of sequences: ', len(self.dataset))

        # plt.figure(1)
        # for f_idx in range(self.img_feature_dim):
        #     plt.subplot(self.img_feature_dim, 1, f_idx + 1)
        #     for grp_idx in range(total_imgft.shape[0]):
        #         plt.plot(total_imgft[grp_idx, :, f_idx], 'gray')
        #     plt.plot(self.avg_imgft[:, f_idx], 'r')
        #
        # plt.figure(2)
        # for f_idx in range(6):
        #     plt.subplot(6, 1, f_idx + 1)
        #     for grp_idx in range(total_jt.shape[0]):
        #         plt.plot(total_jt[grp_idx, :, f_idx], 'gray')
        #     plt.plot(self.avg_jt[:, f_idx], 'r')
        #
        # plt.show()

    def valid_gen_zc_dataset(self, start_steps, step_length, avg_jt, avg_imgft):
        total_imgft = np.empty((0, step_length, self.img_feature_dim))
        total_jt = np.empty((0, step_length, 6))
        for path_idx in range(len(self.img_mot_paths)):
            # images
            path = self.img_mot_paths[path_idx]
            print(path)
            os.chdir(path)
            img_names = sorted(glob.glob('*.jpg'), key=self.numerical_sort)
            img_dataset = TaskImageDataset(img_names, path)

            grp_imgft = np.empty((0, self.img_feature_dim))

            for img_idx in range(start_steps[path_idx], start_steps[path_idx]+step_length):
                img_arr = img_dataset.get_example(img_idx) - self.mean_img.transpose(2, 0, 1)
                with chainer.using_config('train', False):
                    feature = self.img_model.encoder_layers(np.expand_dims(img_arr, axis=0))
                assert feature.size == self.img_feature_dim
                grp_imgft = np.append(grp_imgft, feature.data, axis=0)

            total_imgft = np.append(total_imgft, grp_imgft[np.newaxis, :, :], axis=0)

            # joint angles
            joint_file = os.path.join(path, 'joint_position.txt')
            joint_angles = utils.load_joint_seq(joint_file, start_steps[path_idx], step_length)
            joint_angles = np.array(joint_angles)
            total_jt = np.append(total_jt, joint_angles[np.newaxis, :, :], axis=0)

        # zero center data
        total_imgft = total_imgft - avg_imgft
        total_jt = total_jt - avg_jt

        # concatenate image features and joint angles
        for path_idx in range(len(self.img_mot_paths)):
            for idx in range(0, step_length - self.time_window + 1):
                img_joint = np.ndarray((self.img_mot_dim, self.time_window), dtype=np.float32)
                for seq_idx in range(0, self.time_window):
                    img_joint[0:self.img_feature_dim, seq_idx] = total_imgft[path_idx, idx+seq_idx, :].transpose()
                    img_joint[-6:, seq_idx] = total_jt[path_idx, idx+seq_idx, :].transpose()
                self.dataset.append(img_joint)

        self.num_data = len(self.dataset)
        print('Number of sequences: ', len(self.dataset))

    def get_minibatch_seqs(self, mini_batch_size):
        sequences = np.ndarray((mini_batch_size, self.img_mot_dim, self.time_window), dtype=np.float32)
        for i in range(mini_batch_size):
            batch_idx = random.randint(0, len(self.dataset) - 1)
            sequences[i] = self.dataset[batch_idx]
        return sequences
    
    def get_ordered_seq(self, index):
        sequences = np.ndarray((1, self.img_mot_dim, self.time_window), dtype=np.float32)
        sequences[0] = self.dataset[index]
        return sequences


class ImgMotionTrain(object):
    def __init__(self, model, max_iter, batch_size, save_model_name):
        # network
        self.model = model
        
        # optimizer
        self.optimizer = O.Adam()
        self.optimizer.setup(self.model)
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

        # training parameters
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.save_model_name = save_model_name
        
    def training(self, train_seqs_maker, valid_seqs_maker):
        self.model.train = True
        train_start = time.time()
        loss_sum = cuda.cupy.zeros(1)
        in_batch_iter = 0
        epoch = 0
        epoch_start = time.time()
        
        model_root_name = r'/home/young/URLearning/model_gpu'
        loss_file = ('TrainLoss' + self.save_model_name).rstrip('.model') + '.txt'
        eval_file = ('EvalLoss' + self.save_model_name).rstrip('.model') + '.txt'

        loss_f = open(os.path.join(model_root_name, loss_file), 'w')
        eval_f = open(os.path.join(model_root_name, eval_file), 'w')
        
        # save joint min-max of training dataset
        # np.save(os.path.join(model_root_name, self.save_model_name.rstrip('.model') + 'minmaxJt.npy'), train_seqs_maker.joint_minmax)
        # joint_min = train_seqs_maker.joint_minmax[0, :][np.newaxis, :, np.newaxis]
        # joint_max = train_seqs_maker.joint_minmax[1, :][np.newaxis, :, np.newaxis]
        # joint_range = joint_max - joint_min

        # save joint average and image average
        np.save(os.path.join(model_root_name, self.save_model_name.rstrip('.model') + 'avgJt.npy'), train_seqs_maker.avg_jt)
        np.save(os.path.join(model_root_name, self.save_model_name.rstrip('.model') + 'avgImgFt.npy'), train_seqs_maker.avg_imgft)
        
        img_feature_dim = train_seqs_maker.img_feature_dim

        with chainer.using_config('train', True):
            for iteration in range(self.max_iter):
                batch_start = time.time()
                sequences = train_seqs_maker.get_minibatch_seqs(self.batch_size)
                # normalize joint angles to [-1, 1]
                # sequences[:, -6:, :] = 2 * (sequences[:, -6:, :] - joint_min) / joint_range - 1

                # normalize centered image feature to [-0.05, 0.05]
                sequences[:, :img_feature_dim, :] = 0.1 * (sequences[:, :img_feature_dim, :] + 1) / 2 - 0.05

                self.model.reset_state()
                self.model.zerograds()
                loss = self.compute_loss(sequences, img_feature_dim)
                print('iteration:{} train_loss:{:.04f} iter_elapsed_time:{:.04f} min'.format(iteration,
                                                                                             float(loss.data),
                                                                                             time.time() - batch_start))

                loss_sum += loss.data
                loss.backward()
                self.optimizer.update()
                in_batch_iter += 1

                if self.batch_size * (iteration + 1) // train_seqs_maker.num_data == (epoch + 1):
                    with chainer.using_config('train', False):
                        # validation
                        valid_iter = 0
                        valid_loss_sum = cuda.cupy.zeros(1)
                        while self.batch_size * (valid_iter + 1) // valid_seqs_maker.num_data < 1:
                            valid_seqs = valid_seqs_maker.get_minibatch_seqs(self.batch_size)
                            # normalize
                            # valid_seqs[:, -6:, :] = 2 * (valid_seqs[:, -6:, :] - joint_min) / joint_range - 1

                            # normalize centered image feature to [-0.05, 0.05]
                            valid_seqs[:, :img_feature_dim, :] = 0.1 * (valid_seqs[:, :img_feature_dim, :] + 1) / 2 - 0.05

                            self.model.reset_state()
                            valid_loss = self.compute_loss(valid_seqs, train_seqs_maker.img_feature_dim)
                            valid_loss_sum += valid_loss.data
                            valid_iter += 1

                        # record loss and save model
                        loss_avg = loss_sum / in_batch_iter / train_seqs_maker.time_window
                        valid_loss_avg = valid_loss_sum / valid_iter / valid_seqs_maker.time_window

                        serializers.save_npz(os.path.join(model_root_name, self.save_model_name), self.model)

                        loss_f.write(str(float(loss_avg)) + '\n')
                        eval_f.write(str(float(valid_loss_avg)) + '\n')

                        epoch += 1
                        print('epoch:{:02d} train_loss:{:.04f} val_loss:{:.04f} epoch_elapsed_time:{:.04f} min'.format(epoch, float(loss_avg), float(valid_loss_avg), (time.time() - epoch_start) / 60))

                        loss_sum = cuda.cupy.zeros(1)
                        in_batch_iter = 0
                        epoch_start = time.time()

            serializers.save_npz(os.path.join(model_root_name, self.save_model_name), self.model)

            print('training elapsed_time:{:.04f} hours'.format((time.time()-train_start) / 3600))
    
    def compute_loss(self, sequences, img_feature_dim):
        loss = 0
        b_size, f_size, t_size = sequences.shape
        first_joint = img_feature_dim
        for i in range(t_size - 1):
            x = Variable(cuda.to_gpu(sequences[:, :, i]))
            if self.model.net_idx == 1:
                t = Variable(cuda.to_gpu(sequences[:, :, i+1]))
                loss += self.model(x, t)
            elif self.model.net_idx == 2:
                t = Variable(cuda.to_gpu(sequences[:, first_joint:first_joint+6, i+1]))
                loss += self.model(x, t)
            elif self.model.net_idx == 3:
                t = Variable(cuda.to_gpu(sequences[:, :, i+1]))
                loss += self.model(x, t)
            elif self.model.net_idx == 4:
                t = Variable(cuda.to_gpu(sequences[:, :, i+1]))
                loss += self.model(x, t)
            elif self.model.net_idx == 5:
                t = Variable(cuda.to_gpu(sequences[:, :, i+1]))
                loss += self.model(x, t)
        return loss


def predict_sequence(model, input_seq, dummy):
    b_size, f_size, t_size = input_seq.shape
    model.reset_state()
    for i in range(t_size):
        x = Variable(cuda.cupy.asarray(input_seq[:, :, i:i+1], dtype=np.float32))
        future = model(x, dummy)
    cpu_future = cuda.to_cpu(future.data)
    return cpu_future


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)
    
    root_name = r'/home/young/URLearning'
    img_model_file = os.path.join(root_name, 'model_gpu/GPU_ImgAE45.model')
    train_ij_paths = []
    valid_ij_paths = []

    for grp_idx in range(1, 129):
        path = os.path.join(root_name, 'scoop_dataset6/group' + str(grp_idx))
        if grp_idx % 16 == 0:
            valid_ij_paths.append(path)
        else:
            train_ij_paths.append(path)

    # parameters
    img_feature_dim = 10
    ij_feature_dim = img_feature_dim + 6
    imgnet_idx = 6
    time_window = 70
    step_length = 125

    train_seq_maker = IJSeqMaker(img_model_file, train_ij_paths, imgnet_idx, time_window, img_feature_dim)
    train_start_steps = [0] * len(train_ij_paths)
    train_seq_maker.gen_zc_dataset(train_start_steps, step_length)
    # train_seq_maker.gen_dataset(train_start_steps, step_length)
    valid_seq_maker = IJSeqMaker(img_model_file, valid_ij_paths, imgnet_idx, time_window, img_feature_dim)
    valid_start_steps = [0] * len(valid_ij_paths)
    valid_seq_maker.valid_gen_zc_dataset(valid_start_steps, step_length, train_seq_maker.avg_jt, train_seq_maker.avg_imgft)
    # valid_seq_maker.gen_dataset(valid_start_steps, step_length)

    # set up lstm model
    max_iteration = 20000
    batch_size = 16
    lstm_model_name = 'GPU_ImgMot46.model'
    gpu_device = 0
    cuda.get_device_from_id(gpu_device).use()
    model = ImgMotLSTM(4)
    model.to_gpu()
    model_root_name = os.path.join(root_name, 'model_gpu')

    """
    Train
    """
    trainer = ImgMotionTrain(model, max_iteration, batch_size, lstm_model_name)
    trainer.training(train_seq_maker, valid_seq_maker)