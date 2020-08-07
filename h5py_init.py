import os
import glob
import h5py
import numpy as np
import cv2
from util import *
import sys

'''
Note, these functions will not work without setting up the directories of video frames as shown in get_dir_lists. 
'''

root_drive = '.'

def get_dir_lists(dset):

    '''
    Gets videos directory path list
    Params:
        str dset: dataset to be loaded
    Returns:
        paths to ADL/NA and Fall/Intrusion videos
    '''

    path_abnormal_vids = root_drive + '/Datasets/{}/Fall/Fall*'.format(dset)
    path_normal_vids = root_drive + '/Datasets/{}/NonFall/ADL*'.format(dset)

    # Update paths if some dataset is arranged in some particular way in folders!
    if dset == 'Thermal_Intrusion':
        path_abnormal_vids = root_drive + '/Datasets/{}/Intrusion/Intru*'.format(dset)
        path_normal_vids = root_drive + '/Datasets/{}/NonIntrusion/NA*'.format(dset)
        
    print(path_normal_vids, path_abnormal_vids)

    # glob returns non-ordered list of all pathnames matching a specified pattern
    normal_vids_dir = glob.glob(path_normal_vids)
    abnormal_vids_dir = glob.glob(path_abnormal_vids)

    if len(abnormal_vids_dir) == 0:
        print('no Fall/Intru vids found')
    
    if len(normal_vids_dir) == 0:
        print('no ADL/NA vids found')

    return normal_vids_dir, abnormal_vids_dir


def init_videos(img_width = 64, img_height = 64, raw = False, dset = 'Thermal_Fall'):

    '''

    Creates or overwrites h5py group corresponding to root_path (in body),
    for the h5py file located at
    'H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    For info on h5py: http://docs.h5py.org/en/stable/quick.html#quick

    The h5py group of nested groups is structured as follows:
    
    Processed (or Raw)
        Split_by_video
            ADL1 or NA1
                Data
                    <HDF5 dataset "Data": shape (1397, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (1397,), type "<i4">
            ADL2 or NA2
                Data
                    <HDF5 dataset "Data": shape (3203, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (3203,), type "<i4">
                .
                .
                .

            ADL{N} or NA{N}
                Data
                    <HDF5 dataset "Data": shape (3203, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (3203,), type "<i4">

            Fall1 or Intru1
                Data
                    <HDF5 dataset "Data": shape (49, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (49,), type "<i4">
                .
                .
                .
            Fall{M} or Intru{M}
                Data
                    <HDF5 dataset "Data": shape (49, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (49,), type "<i4">


            where N is number of ADL/NA videos, and M is number of Fall/Intrusion videos.

    Params:
        bool raw: if true, data will be not processed (mean centering and intensity scaling)
        int img_wdith: width of images
        int img_height: height of images
        str dset: dataset to be loaded
    '''

    path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    normal_vids_dir, abnormal_vids_dir = get_dir_lists(dset) # Directories of ADL/NA, Fall/Intrusion videos

    if len(normal_vids_dir) == 0 and len(abnormal_vids_dir) == 0:
        print('no videos found, make sure video files are placed in Datasets folder, terminating...')
        sys.exit()

    if raw == False: 
        root_path = dset + '/Processed/Split_by_video'
    else:
        root_path = dset + '/Raw/Split_by_video'

    print('creating data at root_path', root_path)

    def init_videos_helper(root_path):
            with h5py.File(path, 'a') as hf:
                root = hf.create_group(root_path)

                for vid_dir in abnormal_vids_dir:
                    # Create hf group for each Fall/Intrusion video with frames and labels for frame
                    init_vid(vid_dir=vid_dir, vid_class=1, img_width=img_width,
                             img_height=img_height, hf=root, raw=raw, dset=dset)

                for vid_dir in normal_vids_dir:
                    # Create hf group for each ADL/NA video with frames and labels for frame
                    init_vid(vid_dir=vid_dir, vid_class=0, img_width=img_width,
                             img_height=img_height, hf=root, raw=raw, dset=dset)

    if os.path.isfile(path):
        # If .h5 already exists
        hf = h5py.File(path, 'a')
        if root_path in hf:
            print('video h5py file exists, deleting old group {}, creating new'.format(root_path))
            del hf[root_path]
            hf.close()
            init_videos_helper(root_path)

        else:
            print('File exists, but no group for this data set; initializing..')
            hf.close()

            init_videos_helper(root_path)

    else:
        print('No data file exists yet; initializing')
        init_videos_helper(root_path)


def init_vid(vid_dir = None, vid_class = None, img_width = 64, img_height = 64,
             hf = None, raw = False,  dset = 'Thermal_Fall'):
    '''
    Creates hf group with vid_dir name to put Data and Labels for each frame
    Processes all frames in video in ascending order & fetches labels for Fall/Intrusion vids from csv if asked

    Params:
        str vid_dir: path to vid dir of frames to be initialized
        int vid_class: 1 for Fall/Intrusion, 0 for NonFall/NonIntrusion
        h5py group: group within which new group is nested
    '''

    print('initializing vid at', vid_dir)

    #--Get all frames from a video reading frames in ↑ order & processing each frame
    data = create_img_data_set(fpath=vid_dir, ht=img_height, wd=img_width, raw=raw, sort=True, dset=dset)
    labels = np.zeros(len(data)) # Labels initialized to 0 for each frame

    vid_dir_name = os.path.basename(vid_dir)
    print('vid_dir_name', vid_dir_name) # eg: ADL2 or Fall31 (vid folder name)
    grp = hf.create_group(vid_dir_name) # Create folder name group eg. ADL2 or Fall31

    # If abnormal video, get fall/intrusion labels for each frame
    if vid_class == 1:
        print('setting fall/intrusion start & end')
        abnormality_start, abnormality_stop = get_abnormal_indices(vid_dir_name, dset)
        labels[abnormality_start-1:abnormality_stop] = 1
    
    grp['Labels'] = labels
    grp['Data'] = data

def get_abnormal_indices(dir_name, dset):

    # Get Fall/Intrusion start, end indices from Labels.csv

    root_dir = './Datasets/'
    labels_dir = root_dir + '/{}/Labels.csv'.format(dset)

    import pandas as pd
    my_data = pd.read_csv(labels_dir, sep=',', header=0, index_col=0)
    start, stop = my_data.loc[dir_name][:2]
    print('start, stop', start, stop)

    return int(start), int(stop)
        

def sort_frames(frames, dset):

        # Frames are organized as: FALL_1-0001.jpg, FALL_1-0003.jpg, FALL_1-0002.jpg,...
        # Sorts the list frames in the ascending order acc to dset type

        print('sorting frames...')
        try:
            frames = sorted(frames, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        except ValueError:
            print('failed to sort vid frames')
            return

        return frames

def create_img_data_set(fpath, ht = 64, wd = 64, raw = False, sort = True, dset = 'Thermal_Fall'):

        '''
        Creates data set of all images located at fpath. Sorts images if asked!

        Params:
            str fpath: path to images to be processed
            bool raw: if True does mean centering and rescaling 
            bool sort: if True, sorts frames, ie. keeps sequential order, which may be lost due to glob
            dset: dataset

        Returns:
            ndarray data: Numpy array of images at fpath. Shape (samples, img_width*img_height),
            samples is number of images at fpath.

        '''

        fpath = fpath.replace('\\', '/')
        # Get all frames inside folder or folders with jpg or png extension
        frames = glob.glob(fpath+'/*.jpg') + glob.glob(fpath+'/*.png')

        if sort == True:
            frames = sort_frames(frames, dset)

        data = np.zeros((frames.__len__(), ht, wd, 1)) # (frame_length, 64, 64, 1)

        for x, i in zip(frames, range(0, frames.__len__())):

            img = cv2.imread(x, 0) #Use this for RGB to GS
            #img = cv2.imread(x,-1) #Use this for loading as is(ie. 16 bit needs this, else gets converted to 8)

            img = cv2.resize(img, (ht, wd)) # Resize image from 480x640 to 64x64
            img = img.reshape(ht, wd, 1) # Reshape image to 64x64x1

            if raw == False:
                # Image Processing
                img = img - np.mean(img) # Mean Centering
                img = img.astype('float32') / 255. # Rescaling
            data[i, :, :, :] = img

        print('data.shape', data.shape)
        return data

def init_data_by_class(vid_class = 'NonFall', dset = 'Thermal',\
        raw = False, img_width = 64, img_height = 64, use_cropped = False): 

    '''
    Creates or overwrites h5py group corresponding to root_path (in body),
    for the h5py file located at
    'N:/FallDetection/Datasets/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    Creates the following structure:

    Processed or Raw
        Split_by_class
            NonFall
                Data
                    <HDF5 dataset "Data": shape (22116, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (22116,), type "<i4">
            Fall
                Data
                    <HDF5 dataset "Data": shape (22116, 4096), type "<f8">
                Labels
                    <HDF5 dataset "Labels": shape (22116,), type "<i4">
    '''

    ht, wd = img_width, img_height

    if dset == 'Thermal' or dset == 'Thermal-Intrusion':
        if vid_class == 'NonFall':
            fpath= root_drive + '/Datasets/{}/{}/ADL*'.format(dset, vid_class)
        elif vid_class == 'Fall':
            fpath= root_drive + '/Datasets/{}/{}/Fall*'.format(dset, vid_class)
        else:
            print('invalid vid class') 
            return

    elif dset == 'UR-Filled':
        if vid_class == 'NonFall':
            fpath= root_drive + '/Datasets/UR_Kinect/{}/filled/adl*'.format(vid_class)
        else:
            fpath= root_drive + '/Datasets/UR_Kinect/{}/filled/Fall*'.format(vid_class)

    elif dset == 'UR':
        if vid_class == 'NonFall':
            fpath= root_drive + '/Datasets/UR_Kinect/{}/original/adl*'.format(vid_class)
        else:
            fpath= root_drive + '/Datasets/UR_Kinect/{}/original/Fall*'.format(vid_class)

    elif dset == 'SDU':
        fpath = root_drive + '/Datasets/SDUFall/{}/ADL*/Depth'.format(vid_class)

    elif dset == 'SDU-Filled':
        fpath = root_drive + '/SDUFall/{}/ADL*/Filled'.format(vid_class)

    data = create_img_data_set(fpath, ht, wd, raw, False) #Don't need to sort? I think there is!

    if data.shape[0] == 0:
        print('no data found, make sure video files are placed in Datasets folder, terminating')
        sys.exit()

    path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    # root path is for h5py tree
    if raw == False: 
        root_path = dset + '/Processed/Split_by_class/'+ vid_class
    else:
        root_path = dset + '/Raw/Split_by_class/'+ vid_class

    if vid_class == 'NonFall':
        labels = np.array([0] * len(data))
    else:
        labels = np.array([1] * len(data))

    with h5py.File(path, 'a') as hf:
        print('creating data at ', root_path)
        if root_path in hf:
            print('root_path {} found, clearing'.format(root_path))
            del hf[root_path]
        root = hf.create_group(root_path)
        root['Data'] = data
        root['Labels'] = labels

def flip_windowed_arr(windowed_data):
    """
    windowed_data: of shape (samples, win_len,...)
    
    returns shape len(windowed_data), win_len, flattened_dim)
    Note: Requires openCV
    """
    win_len = windowed_data.shape[1]
    flattened_dim = np.prod(windowed_data.shape[2:])
    #print(flattened_dim)
    flipped_data_windowed = np.zeros((len(windowed_data), win_len, flattened_dim)) #Array of windows
    print(flipped_data_windowed.shape)
    i=0
    for win_idx in range(len(windowed_data)):
        window = windowed_data[win_idx]
        flip_win = np.zeros((win_len, flattened_dim))

        for im_idx in range(len(window)):
            im = window[im_idx]
            hor_flip_im = cv2.flip(im,1)
            #print(hor_flip_im.shape)
            #print(flip_win[im_idx].shape)
            
            flip_win[im_idx] = hor_flip_im.reshape(flattened_dim)
            
        flipped_data_windowed[win_idx] = flip_win
    return flipped_data_windowed

