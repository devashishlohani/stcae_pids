from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, f1_score, auc, \
    precision_recall_curve
import glob
import os
import numpy as np
import pickle
# from pathlib import Path
from sklearn.utils import class_weight as cw
# import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from natsort import natsorted

import cv2
import h5py
from random import randint, seed
from imblearn.metrics import geometric_mean_score
import re
import sys
import pandas as pd
from data_management import *


def threshold(predictions=None, t=0.5):
    # Threshold each frame prediction and assigns it a class
    temp = predictions.copy()
    predicted_classes = temp.reshape(predictions.shape[0])
    for i in range(len(predicted_classes)):
        if predicted_classes[i] < t:
            predicted_classes[i] = 0
        else:
            predicted_classes[i] = 1

    return predicted_classes


def get_output(labels, predictions, get_thres=False, to_plot=False, data_option=None, t=0.5, pos_label=1,
               dset='Thermal', model_name='dst', dir_name='None'):
    # Calculate Confusion matrix, AUROC, AUPR and Geometric mean score!

    predicted_classes = threshold(predictions, t)  # not useful t=0.5
    true_classes = labels

    # create confusion matrix (2 classes)
    conf_mat = confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
    # report = classification_report(true_classes, predicted_classes)
    g_mean = geometric_mean_score(labels, predicted_classes)
    AUROC = []
    AUPR = []

    if np.count_nonzero(labels) > 0 and np.count_nonzero(labels) != labels.shape[0]:
        # Makes sure both classes present

        fpr, tpr, thresholds_roc = roc_curve(y_true=true_classes, y_score=predictions,
                                             pos_label=pos_label)
        # thresholds_roc is from 1 to min. value of prediction i.e. x-mean or x-std score

        AUROC = auc(fpr, tpr)

        precision, recall, thresholds_pr = precision_recall_curve(true_classes,
                                                                  predictions)
        # thresholds_pr is from min. value of prediction to  max value
        # precision goes towards 1 and recall from 1 to  0

        AUPR = auc(recall, precision)

        precision = precision[:-1]  # TODO verify this
        recall = recall[:-1]

        if get_thres:
            optimal_roc_threshold, fpr_thres, tpr_thres = get_roc_optimal_threshold(tpr, fpr, thresholds_roc)
            optimal_pr_threshold, recal_thres, prec_thres = get_pr_optimal_threshold(precision, recall, thresholds_pr)
            if to_plot == True:
                score_dir = './AEComparisons/individual_scores/{}'.format(dset)
                score_dir = score_dir + '/{}/{}'.format(model_name, dir_name)
                if not os.path.isdir(score_dir):
                    os.makedirs(score_dir)
                print('saving ind. scores to {}'.format(score_dir))

                # plot_fpr_tpr(thresholds_roc, tpr, fpr, [fpr_thres, tpr_thres, optimal_roc_threshold], data_option, score_dir, True )
                #
                # plot_ROC_AUC(fpr, tpr, AUROC, data_option, thresholds_roc,
                #              [fpr_thres, tpr_thres, optimal_roc_threshold], save_fig=True, save_dir = score_dir)

                # plot_ROC_AUC_3D(fpr, tpr, AUROC, data_option, thresholds_roc,
                #              [fpr_thres, tpr_thres, optimal_roc_threshold],save_pickle=False, save_dir=score_dir)

                # no_skill = len(true_classes[true_classes==1]) / len(true_classes)
                # plot_PR_AUC(recall, precision, AUPR, no_skill, data_option, thresholds_pr,
                #             [recal_thres, prec_thres, optimal_pr_threshold], save_fig=True, save_pickle=True, save_dir = score_dir)
                #
                plot_RE_hist(labels, predictions, data_option, dir_name, save_dir=score_dir, save_fig=True)
            return AUROC, conf_mat, g_mean, AUPR, optimal_roc_threshold, optimal_pr_threshold
    else:
        print('only one class present')

    return AUROC, conf_mat, g_mean, AUPR


def get_roc_optimal_threshold(tpr, fpr, thresholds):
    ####################################
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    # returns thrshold and associated FPR(to plot)
    ####################################

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i),
                        'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = list(roc_t['threshold'])
    return threshold[0], list(roc_t['fpr'])[0], list(roc_t['tpr'])[0]


def get_pr_optimal_threshold(precision, recall, thresholds):
    ####################################
    # The optimal cut off would be where both precision & recall are high
    # precision - recall is zero or near to zero is the optimal cut off point
    # returns thrshold and associated recall(to plot)
    ####################################

    i = np.arange(len(thresholds))
    roc = pd.DataFrame({'tf': pd.Series(precision - recall, index=i),
                        'threshold': pd.Series(thresholds, index=i),
                        'recall': pd.Series(recall, index=i),
                        'precision': pd.Series(precision, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = list(roc_t['threshold'])
    return threshold[0], list(roc_t['recall'])[0], list(roc_t['precision'])[0]


def plot_ROC_AUC(fpr, tpr, roc_auc, data_option, thresholds=[], optimal_point=[0, 0, 0], save_fig=False,
                 save_pickle=False, save_dir="./AEComparisons/individual_scores/Thermal/DSTC"):
    fig = plt.figure(figsize=(8, 5))
    lw = 2
    l1, = plt.plot(fpr, tpr, color='darkorange', lw=lw)  # , label=)
    l2, = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # , label='No Skill')
    for x, y in zip(fpr, tpr):
        plt.scatter(x, y, c='black')
    plt.plot(optimal_point[0], optimal_point[1], 'go')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    plt.legend([l1, l2], ["ROC curve", "No Skill"], loc='lower right')
    plt.title('ROC for {}, AUROC: {}, opt_threshold: {}'.format(data_option,
                                                                round(roc_auc, 4),
                                                                round(optimal_point[2], 5)),
              fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig != False:
        plt.savefig(save_dir + '/' + data_option + '-roc_curve.png')
    if save_pickle != False:
        with open(save_dir + '/' + data_option + '-roc_curve.pickle', 'wb') as file:
            pickle.dump(fig)
    plt.close()


def plot_ROC_AUC_3D(fpr, tpr, roc_auc, data_option, thresholds=[], optimal_point=[0, 0, 0], save_pickle=False,
                    save_dir="./AEComparisons/individual_scores/Thermal/DSTC"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 5))
    lw = 2
    ax = fig.gca(projection='3d')
    ax.plot(thresholds[1:], fpr[1:], tpr[1:], label='ROC curve', lw=lw, color='darkorange')
    ax.plot([thresholds[-1], thresholds[-1]], [0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='No Skill')
    ax.set_xlim([thresholds[-1], thresholds[1]])
    ax.set_ylim([1, 0])
    ax.set_zlim([0.0, 1.05])
    ax.set_xlabel('Threshold', fontsize=14)
    ax.set_ylabel('FPR', fontsize=14)
    ax.set_zlabel('TPR', fontsize=14)
    for x, y, z in zip(thresholds[1:], fpr[1:], tpr[1:]):
        ax.scatter(x, y, z, c='black')
    ax.legend(loc='right')
    plt.title('ROC for {}, AUROC: {}, opt_threshold: {}'.format(data_option,
                                                                round(roc_auc, 4),
                                                                round(optimal_point[2], 5)),
              fontsize=15)
    plt.show()
    if save_pickle != False:
        with open(save_dir + '/' + data_option + '-roc_curve_3D.pickle', 'wb') as file:
            pickle.dump(fig, file)
    plt.close()


def plot_PR_AUC(recall, precision, pr_auc, no_skill, data_option, thresholds=[], optimal_point=[0, 0, 0],
                save_fig=False, save_pickle=False, save_dir="./AEComparisons/individual_scores/Thermal/DSTC"):
    fig = plt.figure(figsize=(8, 5))
    lw = 2
    l1, = plt.plot(recall, precision, color='darkorange', lw=lw)
    l2, = plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=lw, linestyle='--')
    for x, y in zip(recall, precision):
        plt.scatter(x, y, c='black')
    plt.plot(optimal_point[0], optimal_point[1], 'go')
    plt.xlim([0.0, 1.0])

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)

    plt.legend([l1, l2], ["PR curve", "No Skill"], loc='upper right')
    plt.title('PR for {}, AUPR: {}, opt_threshold: {}'.format(data_option,
                                                              round(pr_auc, 4),
                                                              round(optimal_point[2], 5)),
              fontsize=15)

    if save_fig != False:
        plt.savefig(save_dir + '/' + data_option + '-pr_curve.png')

    if save_pickle != False:
        plt.ylim([0.0, 1.05])
        with open(save_dir + '/' + data_option + '-pr_curve.pickle', 'wb') as file:
            pickle.dump(fig, file)

    # plt.show()
    plt.close()


def plot_RE_hist(labels, predictions, data_option, dir_name, save_dir="", save_fig=False):
    non_fall_indices = np.nonzero(labels == 0)[0]
    non_fall_preds = predictions[non_fall_indices]
    fall_preds = np.delete(predictions, non_fall_indices)
    fig = plt.figure(figsize=(8, 5))
    if fall_preds.size != 0:
        plt.hist(fall_preds, bins='auto', histtype="step", lw=2, color="red", label="Intrusion")
    if non_fall_preds.size != 0:
        plt.hist(non_fall_preds, bins='auto', histtype="step", lw=2, color="blue", label="Non-Intrusion")
    plt.xlabel('RE Score {}'.format(data_option), fontsize=14)
    plt.ylabel('Frames', fontsize=14)
    plt.title('{} Intrusion Frames: {}, Non-Intrusion: {}, Total: {}'.
              format(dir_name, len(fall_preds), len(non_fall_preds), len(predictions)), fontsize=15)
    plt.legend()

    if save_fig != False:
        plt.savefig(save_dir + '/' + data_option + '-hist.png')

    '''
    fall_mean = np.mean(fall_preds)
    fall_std = np.std(fall_preds)
    non_fall_mean = np.mean(non_fall_preds)
    non_fall_std = np.std(non_fall_preds)
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    
    fig = plt.figure(figsize=(8, 5))
    plt.hist(predictions, bins='auto', histtype="step", lw=2)
    plt.xlabel('RE Score {}'.format(data_option), fontsize=14)
    plt.ylabel('Frames', fontsize=14)
    plt.title('Histogram for {}, Total Frames: {}'.format(dir_name, len(predictions)), fontsize=15)
    '''


def plot_fpr_tpr(thresholds, tpr, fpr, optimal_point, data_option, save_dir="", save_fig=False):
    fig = plt.figure(figsize=(8, 5))
    lw = 2
    l1, = plt.plot(thresholds, tpr, color='green', lw=lw)
    plt.scatter(thresholds, tpr, c='black')
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    ax2 = plt.gca().twinx()
    l2, = ax2.plot(thresholds, fpr, color='red', lw=lw)
    ax2.scatter(thresholds, fpr, c='black')
    l3 = ax2.axvline(optimal_point[2], 0, 1, lw=lw)
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([thresholds[-1], thresholds[1] + (thresholds[-2] - thresholds[-1])])
    ax2.set_ylabel('False Positive Rate', fontsize=14)
    plt.legend([l1, l2, l3], ["TPR", "FPR", "Optimal Threshold"],
               loc='top right')
    plt.title('TPR, FPR w.r.t. Threshold ({}), opt_threshold: {}'.format(data_option, round(optimal_point[2], 5)),
              fontsize=15)

    if save_fig != False:
        plt.savefig(save_dir + '/' + data_option + '-FPRvTPR.png')


def MSE(y, t):
    '''
    Mean sqaured error
    '''
    y, t = y.reshape(len(y), np.prod(y.shape[1:])), t.reshape(len(t), np.prod(t.shape[1:]))
    return np.mean(np.power(y - t, 2), axis=1)


def plot_MSE_per_sample(test_data, test_data_re, show=True, marker='o-', label='label'):
    print('test_data.shape', test_data.shape)
    recons_error = MSE(test_data, test_data_re)
    print('recons_error.mean()', recons_error.mean())

    plt.plot(recons_error, marker, label=label)
    if show == True:
        plt.show()
    if label != None:
        plt.legend()


def plot_MSE_per_sample_conv(y, t):
    mse = np.zeros(len(y))
    for i in np.arange(len(y)):
        mse[i] = calc_mse_conv(y[i], t[i])
    print('mse.mean()', mse.mean())
    plt.plot(mse, 'o-')
    plt.show()


def play_frames(frames, decoded_frames=[], labels=[]):
    ht, wd = 64, 64  # TODO change to frames.shape...
    for i in range(len(frames)):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)

        if len(labels) > 0:

            cv2.namedWindow('labels', cv2.WINDOW_NORMAL)
            if labels[i] == 1:
                cv2.imshow('labels', 255 * np.ones((ht, wd)))
            else:
                cv2.imshow('labels', np.zeros((ht, wd)))

        cv2.imshow('image', frames[i].reshape(ht, wd))

        if len(decoded_frames) > 0:
            cv2.namedWindow('decoded', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('decoded', 600, 600)
            cv2.imshow('decoded', decoded_frames[i].reshape(ht, wd))

        cv2.waitKey(10)
    cv2.destroyAllWindows()


# init_videos(img_width = 64, img_height = 64, data_option = 'Option1')

def generate_test_vid_names(data_dict, dset):

    ##--generates test folder names

    if dset == 'Thermal_Fall':
        vid_base_name = 'Fall'
    elif dset == 'Thermal_Intrusion':
        vid_base_name = 'Intru'

    return natsorted([x for x in list(data_dict.keys()) if vid_base_name in x])

def generate_vid_keys(vid_base_name, dset):
    # Generates list of video folder names acc to video class
    # Eg: for Thermal Fall: [Fall1, Fall2,..,Fall35]

    if dset == 'IDD':
        if vid_base_name == 'Intru' or vid_base_name == 'NFFall':
            num_vids = 150  # 120, 17, 10
        elif vid_base_name == 'NA':
            num_vids = 80 # 50 , 20
        else:
            print('invalid basename')

    if dset == 'Thermal-Intrusion':
        if vid_base_name == 'Fall' or vid_base_name == 'NFFall':
            num_vids = 31  # 17
        elif vid_base_name == 'ADL':
            num_vids = 6  # 20
        else:
            print('invalid basename')

    if dset == 'Thermal' or dset=='Thermal_Fall':
        if vid_base_name == 'Fall' or vid_base_name == 'NFFall':
            num_vids = 35
        elif vid_base_name == 'ADL':
            num_vids = 9
        else:
            print('invalid basename')

    if dset == 'Thermal-Dummy':
        if vid_base_name == 'Fall' or vid_base_name == 'NFFall':
            num_vids = 2
        elif vid_base_name == 'ADL':
            num_vids = 2
        else:
            print('invalid basename')

    elif dset == 'UR' or dset == 'UR-Filled':
        if vid_base_name == 'Fall' or vid_base_name == 'NFFall':
            num_vids = 30
        elif vid_base_name == 'ADL':
            num_vids = 40

        else:
            print('invalid basename')

    elif dset == 'TST':
        if vid_base_name == 'Fall' or vid_base_name == 'NFFall':
            num_vids = 80  # TODO update to 132 once init
        elif vid_base_name == 'ADL':
            num_vids = 132
        else:
            print('invalid basename')

    elif dset == 'SDU' or dset == 'SDU-Filled':
        if vid_base_name == 'Fall' or vid_base_name == 'NFFall':
            num_vids = 200  # TODO update to 132 once init
        elif vid_base_name == 'ADL':
            num_vids = 1000
        else:
            print('invalid basename')

    if (dset == 'UR' or dset == 'UR-Filled') and vid_base_name == 'ADL':
        keys = ['adl-{num:02d}-cam0-d'.format(num=i + 1) for i in range(num_vids)]
    else:
        keys = [vid_base_name + str(i + 1) for i in range(num_vids)]

    return keys


def plot_ROC_AUC_tol(fpr, tpr, roc_auc, data_option, tolerance):
    '''
    plots fo rmultiple tolerance
    '''

    # plt.figure()
    lw = 2
    plt.plot(fpr, tpr, \
             lw=lw, label='tolerance %0.1f (area = %0.4f)' % (tolerance, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {}'.format(data_option))
    plt.legend(loc="lower right")
    # plt.close()
    # plt.show()

    return plt


def make_cross_window_matrix(scores):
    """
    Takes input of form (windows, window_length)
    and creates matrix of form (image_index, cross_window_score)
    """

    win_len = scores.shape[1]  # 8 in our case
    mat = np.zeros((len(scores) + win_len - 1, len(scores)))  # (#images, windows)
    mat[:] = np.NAN

    for i in range(len(scores)):
        win = scores[i]  # Get RE matrix from window i
        mat[i:len(win) + i, i] = win  # Put in img matrx (corresponding to frames the window effect)

    return mat

def get_RE_R(scores):
    """
    Returns simple RE, the RE per frame
    Takes input of form (windows, window_length)
    and output of RE(list) for each frame
    """

    win_len = scores.shape[1]  # 8 in our case
    R = np.zeros(len(scores) + win_len - 1) # R for each frame of video

    R[:win_len] = scores[0] # Get RE of first window(8 frames in our case) for first 8 frames
    for i in range(1, len(scores)):
        R[win_len + i - 1] = scores[i][-1] # get the last frame RE from current window and update R

    return R


def get_cross_window_stats(scores_mat):
    '''
    calculate cross context scores from cross window matrix
    Assumes scores in form (image_index,cross_window_scores), ie. shape (frames,window_len)
    returns in form (img_index, mean, std, mean+std)
    '''
    scores_final = []
    for i in range(len(scores_mat)):
        row = scores_mat[i, :]  # Take all scores of an img in different windows
        mean = np.nanmean(row, axis=0)
        std = np.nanstd(row, axis=0)
        scores_final.append((mean, std, mean + std * 10 ** 3))

    print(len(scores_final))
    scores_final = np.array(scores_final)
    return scores_final

def get_cross_window_mean(scores_mat):
    '''
    calculate cross context mean score from cross window matrix
    Assumes scores in form (image_index,cross_window_scores), ie. shape (frames,window_len)
    returns in form (img_index, mean)
    '''
    scores_final = []
    for i in range(len(scores_mat)):
        row = scores_mat[i, :]  # Take all scores of an img in different windows
        mean = np.nanmean(row, axis=0)
        scores_final.append(mean)

    print(len(scores_final))
    scores_final = np.array(scores_final)
    return scores_final

def get_cross_window_std(scores_mat):
    '''
    calculate cross context std score from cross window matrix
    Assumes scores in form (image_index,cross_window_scores), ie. shape (frames,window_len)
    returns in form (img_index, std)
    '''
    scores_final = []
    for i in range(len(scores_mat)):
        row = scores_mat[i, :]  # Take all scores of an img in different windows
        std = np.nanstd(row, axis=0)
        scores_final.append(std)

    print(len(scores_final))
    scores_final = np.array(scores_final)
    return scores_final

def agg_window(RE, agg_type):
    '''
    Aggregates window of scores in various ways
    Input RE: (#windows, win_len)
    '''

    # Within Context(window) Anomaly Score
    # o/p shape: (#windows, ), i.e. score per window
    if agg_type == 'in_mean':
        inwin_mean = np.mean(RE, axis=1)
        return inwin_mean

    elif agg_type == 'in_std':
        inwin_std = np.std(RE, axis=1)
        return inwin_std

    # Cross Context Anomaly Scores
    elif agg_type == 'x_mean':
        RE_xmat = make_cross_window_matrix(RE)
        #r stats = get_cross_window_stats(RE_xmat)
        #r x_mean = stats[:, 0]
        x_mean = get_cross_window_mean(RE_xmat)
        return x_mean

    elif agg_type == 'x_std':
        RE_xmat = make_cross_window_matrix(RE)
        #r stats = get_cross_window_stats(RE_xmat)
        #r x_std = stats[:, 1]
        x_std = get_cross_window_std(RE_xmat)
        return x_std

    elif agg_type == 'r':
        R = get_RE_R(RE)
        return R
    else:
        print('agg_type not found')


def restore_Fall_vid(data_dict, Fall_name, NFF_name):
    fall_start = data_dict[Fall_name + '/Data'].attrs[
        'Fall start index']  # Restores sequence order, experiment.use_cropped != data.use_cropped always

    fall_start -= 1

    Fall_data, Fall_labels = data_dict[Fall_name + '/Data'][:], data_dict[Fall_name + '/Labels'][:]
    NFF_data, NFF_labels = data_dict[NFF_name + '/Data'][:], data_dict[NFF_name + '/Labels'][:]
    vid_total = np.concatenate((NFF_data[:fall_start], Fall_data, NFF_data[fall_start:]), axis=0)
    labels_total = np.concatenate((NFF_labels[:fall_start], Fall_labels, NFF_labels[fall_start:]), axis=0)

    return vid_total, labels_total


def get_thresholds_helper(RE, omega=1.5):
    '''
        Gets all threshodls from RE

        Params:
            ndarray RE: reconstruction error of training data
        '''

    Q_3, Q_1 = np.percentile(RE, [75, 25])
    IQR = Q_3 - Q_1
    # omega = 1.5
    RRE = RE[(Q_1 - omega * IQR <= RE) & (RE <= Q_3 + omega * IQR)]

    t1, t2, t3, t4, t5, t6 = np.mean(RE), np.mean(RE) + np.std(RE), np.mean(RE) + 2 * np.std(RE), np.mean(
        RE) + 3 * np.std(RE), np.max(RE), np.max(RRE)
    thresholds = [t1, t2, t3, t4, t5, t6]

    return thresholds


def animate_fall_detect_Spresent(testfall, recons, error_map_seq, scores, win_len=1, threshold=0, to_save='./test.mp4',
                                 legend_label='RRE'):
    '''
    Pass in data for single video, recons is recons frames,  etc.
    Threshold is RRE, mean, etc..
    testfall: test video in frames
    recons: windowed recons!!
    scores: x_std or x_mean
    '''
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

    ht, wd = 64, 64

    eps = .0001
    # setup figure
    # fig = plt.figure()
    fig, ((ax1, ax3, ax4)) = plt.subplots(1, 3, figsize=(8, 8))

    ax1.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    # ax1=fig.add_subplot(2,2,1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set_title("Reconstruction")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_title("Error")
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_ylabel('Reconstruction Error '+ legend_label)
    ax2.set_xlabel('Frame')

    if threshold != 0:
        ax2.axhline(y=threshold, color='b', linestyle='dashed', label="Threshold")
        ax2.legend()
        ax2.set_title("Threshold: {}".format(round(threshold, 6)))

    # set up list of images for animation
    ims = []

    for time in range(len(testfall) - (win_len - 1)):

        im1 = ax1.imshow(testfall[time].reshape(ht, wd), cmap='gray', aspect='equal')
        im2 = ax3.imshow(recons[time].reshape(ht, wd), cmap='gray', aspect='equal')
        im3 = ax4.imshow(error_map_seq[time].reshape(ht, wd), cmap='gray', aspect='equal')

        # print("time={} mse={} std={}".format(time,mse_difficult[time],std))
        if time > 0:

            scores_curr = scores[0:time]

            fall_pts_idx = np.argwhere(scores_curr > threshold)
            nonfall_pts_idx = np.argwhere(scores_curr <= threshold)

            fall_pts = scores_curr[fall_pts_idx]
            nonfall_pts = scores_curr[nonfall_pts_idx]

            if fall_pts_idx.shape[0] > 0:
                # pass
                plot_r, = ax2.plot(fall_pts_idx, fall_pts, 'r.') # chnage here for nomral color r.
                plot, = ax2.plot(nonfall_pts_idx, nonfall_pts, 'g.')
            else:

                plot, = ax2.plot(scores_curr, 'g.')

        else:
            plot, = ax2.plot(scores[0], 'g.')
            plot_r, = ax2.plot(scores[0], 'g.')

        ims.append([im1, plot, im2, plot_r, im3])  # list of ims

    # run animation
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat=False)
    # plt.tight_layout()
    gs.tight_layout(fig)
    ani.save(to_save)

    ani.event_source.stop()
    del ani
    plt.close()
    # plt.show()
    # return ani

def animate_fall_detect_Spresent_o(testfall, recons, error_map_seq, scores, win_len=1, threshold=0, to_save='./test.mp4',
                                 legend_label='RRE'):
    '''
    Pass in data for single video, recons is recons frames,  etc.
    Threshold is RRE, mean, etc..
    testfall: test video in frames
    recons: windowed recons!!
    scores: x_std or x_mean
    '''
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    ht, wd = 64, 64

    eps = .0001
    fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(8, 8))

    ax1.axis('off')
    ax3.axis('off')

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set_title("Reconstruction")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_ylabel('RE Error (R)')
    ax2.set_xlabel('Frame')

    if threshold != 0:
        ax2.axhline(y=threshold, color='b', linestyle='dashed', label=legend_label)
        ax2.legend()
        ax2.set_title("Threshold: {}".format(round(threshold, 6)))

    # set up list of images for animation
    ims = []

    for time in range(len(testfall) - (win_len - 1)):

        im1 = ax1.imshow(testfall[time].reshape(ht, wd), cmap='gray', aspect='equal')
        im2 = ax3.imshow(recons[time].reshape(ht, wd), cmap='gray', aspect='equal')

        # print("time={} mse={} std={}".format(time,mse_difficult[time],std))
        if time > 0:

            scores_curr = scores[0:time]

            fall_pts_idx = np.argwhere(scores_curr > threshold)
            nonfall_pts_idx = np.argwhere(scores_curr <= threshold)

            fall_pts = scores_curr[fall_pts_idx]
            nonfall_pts = scores_curr[nonfall_pts_idx]

            if fall_pts_idx.shape[0] > 0:
                # pass
                plot_r, = ax2.plot(fall_pts_idx, fall_pts, 'r.')
                plot, = ax2.plot(nonfall_pts_idx, nonfall_pts, 'b.')
            else:

                plot, = ax2.plot(scores_curr, 'b.')

        else:
            plot, = ax2.plot(scores[0], 'b.')
            plot_r, = ax2.plot(scores[0], 'b.')

        ims.append([im1, plot, im2, plot_r])  # list of ims

    # run animation
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat=False)
    # plt.tight_layout()
    gs.tight_layout(fig)
    ani.save(to_save)

    ani.event_source.stop()
    del ani
    plt.close()
    # plt.show()
    # return ani

def join_mean_std(mean, std):
    '''
    mean(std) for matrix of means and stds (same size)
    '''

    mean_fl = mean.flatten()
    std_fl = std.flatten()
    new = np.ones(std_fl.shape, dtype=object)

    for i in range(len(std_fl)):
        new[i] = "{:.2f}({:.2f})".format(mean_fl[i], std_fl[i])

    new = np.reshape(new, mean.shape)
    return new


def gather_auc_avg_per_tol(inwin_mean, inwin_std, labels, win_len=8):
    '''
    inwin_mean/std are scores for each window(1 score for all frames) for a video
    (1,num_windows = vid_length - win_len-1)
    labels: gt classes (0 or 1)

    Returns array of shape (2, win_len = tolerance*2), which are scores for each tolerance in
    range(win_len), *2 for std and mean, one row for AUROC, one for AUPR
    tol1_mean, tol1_std, tol2_men, tol2_std.....
    '''

    stride = 1
    tol_mat = np.zeros((2, 2 * win_len))
    tol_list_ROC = []
    tol_list_PR = []

    tol_keys = []  # For dataframe labels
    for tolerance in range(win_len):
        tolerance += 1  # Start at 1

        windowed_labels = create_windowed_labels(labels, stride, tolerance, win_len)  # label for each window

        AUROC_mean, conf_mat, g_mean, AUPR_mean = get_output(labels=windowed_labels, \
                                                             predictions=inwin_mean,
                                                             data_option='inwin-mean', to_plot=False)  # single value

        AUROC_std, conf_mat, g_mean, AUPR_std = get_output(labels=windowed_labels,
                                                           predictions=inwin_std,
                                                           data_option='inwin-std', to_plot=False)

        tol_keys.append('tol_{}-mean'.format(tolerance))  # heading for csv, eg: tol_1-mean
        tol_list_ROC.append(AUROC_mean)  # auroc_mean val associated for this tol_1-mean

        tol_keys.append('tol_{}-std'.format(tolerance))  # heading for csv, eg: tol_1-std
        tol_list_ROC.append(AUROC_std)  # auroc_std val associated for this tol_1-mean

        # append similarly AUPR values (without heading, bcoz only 1 column heading possible
        tol_list_PR.append(AUPR_mean)
        tol_list_PR.append(AUPR_std)

    # Put AUROC and AUPR values for each tolerance into a numpy array
    ROCS = np.array(tol_list_ROC)
    PRS = np.array(tol_list_PR)

    # Put this values in total matrix
    tol_mat[0, :] = ROCS
    tol_mat[1, :] = PRS

    # return tol_matrix & headings
    return tol_mat, tol_keys


def create_windowed_labels(labels, stride, tolerance, window_length):
    '''
    Create labels on seq level: label for each window
    int tolerance: # fall frames (1's) in a window for it to be labeled as a fall (1).
    Must not exceed window length!
    '''

    output_length = int(np.floor((len(labels) - window_length) / stride)) + 1  # #windows
    output_shape = (output_length, 1)
    total = np.zeros(output_shape)  # intialize window labels with 0

    i = 0
    while i < output_length:
        next_chunk = np.array([labels[i + j] for j in range(window_length)])  # Frame labels for a window
        num_falls = sum(next_chunk)  # number of falls in the window

        if num_falls >= tolerance:
            total[i] = 1
        else:
            total[i] = 0

        i = i + stride

    labels_windowed = total
    return labels_windowed


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def show_pickle_fig(pickle_path):
    with open(pickle_path, 'rb') as file:
        figx = pickle.load(file)
        figx.show()

