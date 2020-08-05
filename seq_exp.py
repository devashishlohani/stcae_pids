import multiprocessing
import os
import time
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import load_model
# import tensorflow.data
import h5py
import glob
from sklearn.metrics import average_precision_score
import matplotlib

matplotlib.use('Agg')
import sys

sys.path.insert(0, './animation')
# from plot_video_animation_3D import *

from img_exp import ImgExp
from h5py_init import init_videos, flip_windowed_arr
# from models import *
from util import generate_vid_keys, generate_test_vid_names, agg_window, get_output, gather_auc_avg_per_tol, \
    animate_fall_detect_Spresent, join_mean_std, truncate , plot_RE_hist, animate_fall_detect_Spresent_o

from data_management import init_windowed_arr, create_windowed_arr, generate_training_windows, make_windows_h5_file, get_windowed_data

root_drive = '.'


class SeqExp(ImgExp):
    '''
        A autoencoder experiment based on sequence of images
        Inherits get_thresh, save exp, and variable initialization

        Attributes:
            int win_len: window length, or the number of contigous frames forming a sample 
        '''

    def __init__(self, model=None, model_name=None, misc_save_info=None,
                 batch_size=32, model_type=None, callbacks_list=None,
                 pre_load=None, initial_epoch=0, epochs=1, dset='Thermal',
                 win_len=8, hor_flip=False, img_width=64, img_height=64):

        ImgExp.__init__(self, model=model, img_width=img_width, \
                        img_height=img_height, model_name=model_name, \
                        batch_size=batch_size, model_type=model_type, \
                        pre_load=pre_load, initial_epoch=initial_epoch, \
                        epochs=epochs, hor_flip=hor_flip, dset=dset)

        self.win_len = win_len

    def set_train_data(self, raw=False, mmap_mode=None):  # TODO init windows from h5py if no npData found

        '''
        loads or initializes windowed train data, and sets self.train_data accordingly
        '''

        to_load = root_drive + '/npData/{}/ADL_data-proc-win_{}.npy'.format(self.dset, self.win_len)

        if os.path.isfile(to_load):
            print('npData found, loading..')
            self.train_data = np.load(to_load, mmap_mode=mmap_mode)
        else:
            print('npData not found, initializing..')

            self.train_data = init_windowed_arr(dset=self.dset, ADL_only=True,
                                                win_len=self.win_len,
                                                img_width=self.img_width,
                                                img_height=self.img_height)


        if self.hor_flip == True:
            to_load_flip = './npData/hor_flip-by_window/{}'. \
                format(os.path.basename(to_load))
            data_flip = self.init_flipped_by_win(to_load_flip)

            self.train_data = np.concatenate((self.train_data, data_flip), axis=0)

    def get_h5_file(self, raw=False, mmap_mode=None):  # TODO init windows from h5py if no npData found
        '''
            Creates or overwrites h5py file
            '''

        master_path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(self.dset, self.img_width,
                                                                               self.img_height)
        if not os.path.isfile(master_path):
            print('initializing h5py..')
            init_videos(img_width=self.img_width, img_height=self.img_height, raw=raw, dset=self.dset)

        return master_path

    def train(self, sample_weight=None):

        """
                trains a sequential autoencoder on windowed data(sequences of
                contiguous frames are reconstucted)
                """

        model_name = self.model_name
        base = './Checkpoints/{}'.format(self.dset)
        base_logs = './logs/{}'.format(self.dset)

        if not os.path.isdir(base):
            os.mkdir(base)
        if not os.path.isdir(base_logs):
            os.mkdir(base_logs)

        checkpointer = ModelCheckpoint(filepath=base + '/' + model_name + '-' + \
                                                '{epoch:03d}-{loss:.3f}.hdf5', period=100, verbose=1)
        timestamp = time.time()
        print('./Checkpoints/' + model_name + '-' + '.{epoch:03d}-{loss:.3f}.hdf5')

        csv_logger = CSVLogger(base_logs + '/' + model_name + '-' + 'training-' + str(timestamp) + '.log')
        callbacks_list = [csv_logger, checkpointer]
        print(callbacks_list)
        print(csv_logger)

        self.model.fit(self.train_data, self.train_data, epochs = self.epochs,
                       batch_size = self.batch_size, verbose = 2,
                       callbacks = callbacks_list, sample_weight = sample_weight)
        self.save_exp()

    def init_flipped_by_win(self, to_load_flip):

        if os.path.isfile(to_load_flip):
            data_flip = np.load(to_load_flip)
            data_flip = data_flip.reshape(len(data_flip), self.train_data.shape[1],
                                          self.train_data.shape[2],
                                          self.train_data.shape[3], 1)

            return data_flip

        else:
            print('creating flipped by window data..')
            data_flip = flip_windowed_arr(self.train_data)

            return data_flip

    def get_MSE(self, test_data, agg_type='r_sigma'):
        '''
            MSE for sequential data (video). Uses data chunking with memap for SDU-Filled.
            Assumes windowed
            
            Params:
                ndarray test_data: data used to test model (reconstrcut). Of 
                    shape (samples, window length, img_width, img_height)
                agg_type: how to aggregate windowed scores

            Returns:
                ndarray: Mean squared error between test_data windows and reconstructed windows,
                aggregated.
                This gives (samples,) shape
    
            '''

        img_width, img_height, win_len, model, stride = self.img_width, self.img_height, \
                                                        self.win_len, self.model, 1

        if test_data.shape[1] != win_len:  # Not windowed
            test_data = test_data.reshape(len(test_data), img_width, img_height, 1)
            test_data = create_windowed_arr(test_data, stride, win_len)

       # Do prediction on test windows with learned model
        recons_seq = model.predict(test_data)  # (samples-win_len+1, win_len, wd,ht,1)
        recons_seq_or = recons_seq

        test_data = test_data.reshape(len(test_data), win_len, img_height * img_width)
        recons_seq = recons_seq.reshape(len(recons_seq), win_len, img_height * img_width)

        error = test_data - recons_seq
        # Calculate Reconstruction Error(MSE) for all windows
        RE = np.mean(np.power(error, 2), axis=2)  # (samples-win_len+1,win_len)

        RE = agg_window(RE, agg_type)

        return RE, recons_seq_or, error

    def get_MSE_all_agg(self, test_data):

        """
            Gets MSE for all aggregate types 'r_sigma', 'r_mu', 'in_std', 'in_mean'.

            Params:
                ndarray test_data: data used to test model (reconstruct).
                shape (samples(windows), window length, img_width, img_height, 1)

            Returns:
                dictionary with keys 'r_sigma', 'r_mu', 'in_std', 'in_mean', and values
                ndarrays of shape (samples,)
            """

        img_width, img_height, win_len, model = self.img_width, self.img_height, self.win_len, \
                                                self.model

        # Do prediction on test windows_animation with learned model
        recons_seq = model.predict(test_data)  # (samples(or frames)-win_len+1, win_len, wd, ht, 1)
        recons_seq_or = recons_seq

        # reshape test windows & reconstructed windows
        test_data = test_data.reshape(len(test_data), win_len, img_height * img_width)
        recons_seq = recons_seq.reshape(len(recons_seq), win_len, img_height * img_width)

        error = test_data - recons_seq

        # Calculate Reconstruction Error(MSE) for all windows
        RE = np.mean(np.power(error, 2), axis=2)  # (samples-win_len+1, win_len)

        RE_dict = {}
        agg_type_list = ['r_sigma', 'r_mu', 'in_std', 'in_mean', 'r']

        # Get various per frame RE score
        for agg_type in agg_type_list:
            RE_dict[agg_type] = agg_window(RE, agg_type)

        return RE_dict, recons_seq_or, error

    def test(self, eval_type = 'per_video', RE_type ='r', animate=False, indicative_threshold_animation=False):

        '''
            Gets AUC ROC/PR for all videos, using various (20) scoring schemes.
            Save scores to './AEComparisons/all_scores/self.dset/self.model_name.csv'
            Assumes self.model has been initialized
            '''

        dset, to_load, img_width, img_height = self.dset, self.pre_load, self.img_width, self.img_height
        stride = 1
        win_len = self.win_len
        model_name = os.path.basename(to_load).split('.')[0]
        print(model_name)

        labels_total_l = []
        vid_index = 0
        preds_total = []

        # h5 file path!
        path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

        # Create h5 if does not already exist
        if not os.path.isfile(path):
            print('initializing h5py..')
            init_videos(img_width=img_width, img_height=img_height, raw=False, dset=dset)

        hf = h5py.File(path, 'r')
        data_dict = hf['{}/Processed/Split_by_video'.format(dset)] # Get into Split by Videos for Fall Folders Data
        # Fall(Test) video folders names [Fall1, Fall2, FallN] without data info!!
        vid_dir_keys = generate_test_vid_names(data_dict, dset)
        #vid_dir_keys_Fall = generate_vid_keys('Fall', dset=dset) # For fall
        #vid_dir_keys_Fall = generate_vid_keys('Intru', dset=dset) # for INtru

        num_vids = len(vid_dir_keys)
        print('num_vids', num_vids)

        base = './AEComparisons/scores/{}/'.format(self.dset)

        if not os.path.isdir(base):
            # create dir if it does not exist!
            os.makedirs(base)

        save_path = base+'{}_{}.csv'.format(model_name + "-" + RE_type, eval_type)

        f = open(save_path, 'w')
        with f :
            fnames = ['AUROC'+':'+RE_type+'('+eval_type+')', 'AUPR'+':'+ RE_type+'('+eval_type+')'] # AUROC:r(all_vids), AUROC:r(per_vid)
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writeheader()

            if(eval_type=='per_video'):
                ROC_mat = np.ones((num_vids, 1))
                PR_mat = np.ones((num_vids, 1))

            for Fall_name in vid_dir_keys:

                print(Fall_name)
                start_time = time.time()
                # Get frames and labels of a Fallx folder
                vid_total = data_dict[Fall_name]['Data'][:]
                labels_total = data_dict[Fall_name]['Labels'][:]
                test_labels = labels_total
                test_data = vid_total.reshape(len(vid_total), img_width, img_height, 1)
                # Create array of frame windows!
                test_data_windowed = create_windowed_arr(test_data, stride, win_len)

                RE_pred, recons_seq, error_seq = self.get_MSE(test_data_windowed, agg_type=RE_type)

                if(eval_type=='all_videos'):
                    labels_total_l.extend(labels_total)
                    preds_total.extend(RE_pred)

                elif(eval_type=='per_video'):
                    plot = False
                    # Get AUROC, Confusion matrix, Geometric mean score & AUPR scores
                    auc_roc, conf_mat, g_mean, auc_pr, roc_thres, pr_thres = get_output(
                        labels=test_labels, \
                        predictions=RE_pred, get_thres=True,
                        data_option=RE_type,
                        to_plot=plot, dset=dset,
                        model_name=model_name,
                        dir_name=Fall_name)
                    ROC_mat[vid_index, 0] = auc_roc
                    PR_mat[vid_index, 0] = auc_pr

                    writer.writerow({fnames[0]: truncate(auc_roc, 3), fnames[1]: truncate(auc_pr, 3)})

                    print("AUROC", auc_roc)
                    print("AUPR", auc_pr)

                    print("Without Animation Time %.2f s or %.2f mins" % (time.time() - start_time, (time.time() - start_time) / 60))

                    if animate == True :
                        ani_dir = './Animation/{}'.format(dset)
                        ani_dir = ani_dir + '/{}'.format(model_name)
                        if not os.path.isdir(ani_dir):
                            os.makedirs(ani_dir)
                        ani_dir = ani_dir + '/{}'.format(Fall_name)
                        if not os.path.isdir(ani_dir):
                            os.makedirs(ani_dir)
                        print('saving animation to {}'.format(ani_dir))
                        tr = 0 # by default no indicative threshold
                        if(indicative_threshold_animation== True):
                            tr = roc_thres
                        animate_fall_detect_Spresent(testfall=test_data, \
                                                     recons=recons_seq[:, int(np.floor(win_len / 2)), :],
                                                     error_map_seq=error_seq[:, int(np.floor(win_len / 2)), :],
                                                     scores=RE_pred,
                                                     to_save=ani_dir + '/{}.mp4'.format(RE_type),
                                                     win_len=8,
                                                     legend_label= RE_type,
                                                     threshold= tr
                                                     )

                        print("With Animation Time %.2f s or %.2f mins" % (
                            time.time() - start_time, (time.time() - start_time) / 60))

                else:
                    print("eval_type wrong...")
                    print("possible options: all_videos or per_video")
                    print("exiting...")
                    exit()
                vid_index += 1  # next video

            if(eval_type=='all_videos'):
                labels_total_l = np.asarray(labels_total_l)
                preds_total = np.asarray(preds_total)
                plot= False
                auc_roc, conf_mat, g_mean, auc_pr, roc_thres, pr_thres = get_output(
                    labels=labels_total_l, \
                    predictions=preds_total, get_thres=True,
                    data_option=RE_type,
                    to_plot=plot, dset=dset,
                    model_name=model_name,
                    dir_name=Fall_name)
                writer.writerow({fnames[0]: truncate(auc_roc, 2), fnames[1]: truncate(auc_pr, 2)})
                f.close()

            else:
                print('ROC_mat.shape', ROC_mat.shape)
                AUROC_avg = np.mean(ROC_mat, axis=0)
                AUROC_std = np.std(ROC_mat, axis=0)
                AUROC_avg_std = join_mean_std(AUROC_avg, AUROC_std)

                # Same for PR values
                AUPR_avg = np.mean(PR_mat, axis=0)
                AUPR_std = np.std(PR_mat, axis=0)
                AUPR_avg_std = join_mean_std(AUPR_avg, AUPR_std)

                writer.writerow({fnames[0]: 'Average (Std)'
                                 })

                writer.writerow({fnames[0]: AUROC_avg_std[0],
                                 fnames[1]: AUPR_avg_std[0],
                                 })

                f.close()