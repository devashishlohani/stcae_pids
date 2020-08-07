import threading
import time
import os
import glob
import h5py
import numpy as np
import cv2
from util import *
import sys
from h5py_init import init_videos, init_data_by_class
root_drive = '.'

def get_windowed_data(dset = 'Thermal_Fall', ADL_only = True, win_len = 8, img_width = 64, img_height = 64, avoid_vid =None, save_np=False):

    '''
    Creates windowed version of dset data. avoid a video if required!

    Params:
        str dset: dataset to use
        bool ADL_only: if True, only takes ADL from dataset
        int win_len: how many frames to extract for a sequence

    Returns:
        ndarray vids_win: shape (samples-D, win_len, )
    '''

    master_path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    if not os.path.isfile(master_path):
        print('initializing h5py..')
        init_videos(img_width = img_width, img_height = img_height, raw = False, dset = dset)

    with h5py.File(master_path, 'r') as hf:

            data_dict = hf[dset + '/Processed/Split_by_video']
            print(data_dict)
            if ADL_only == True:
                data_dict = dict((key, value) for key, value in data_dict.items() if
                     ('ADL' in key or 'NA' in key) and (key != avoid_vid))
            print(data_dict)
            vids_win = create_windowed_arr_per_vid(vids_dict = data_dict, \
                        stride = 1, \
                        win_len = win_len,\
                        img_width= img_width,\
                        img_height= img_height)

            if ADL_only == True and save_np==True:
                save_path = root_drive + '/npData/{}/'.format(dset)

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                save_path = save_path + 'Avoid' + avoid_vid +'ADL_data-proc-win_{}.npy'.format(win_len)

                print('saving data to ', save_path)
                np.save(save_path, vids_win)

            print('total windowed array shape', vids_win.shape)

    return vids_win

def init_windowed_arr(dset = 'Thermal_Fall', ADL_only = True, win_len = 8, img_width = 64, img_height = 64):

    '''
    Creates windowed version of dset data. Saves windowed array to
    npData folder as npy file

    Params:
        str dset: dataset to use
        bool ADL_only: if True, only takes ADL from dataset
        int win_len: how many frames to extract for a sequence

    Returns:
        ndarray vids_win: shape (samples-D, win_len, )
    '''

    master_path = root_drive + '/H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    if not os.path.isfile(master_path):
        print('initializing h5py..')
        init_videos(img_width=img_width, img_height=img_height, raw=False, dset=dset)

    with h5py.File(master_path, 'r') as hf:

            data_dict = hf[dset + '/Processed/Split_by_video']
            if ADL_only == True:
                # Get only normal behaviour vids
                data_dict = dict((key,value) for key, value in data_dict.items() if 'ADL' in key or 'NA' in key)

            vids_win = create_windowed_arr_per_vid(
                        vids_dict=data_dict,
                        stride=1,
                        win_len=win_len,
                        img_width=img_width,
                        img_height=img_height)

            if ADL_only == True:
                save_path = root_drive + '/npData/{}/'.format(dset)

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                save_path = save_path + 'training_data-imgdim_{}x{}-win_{}.npy'.format(img_width, img_height, win_len)

                print('saving data to ', save_path)
                np.save(save_path, vids_win)

            print('total windowed array shape', vids_win.shape)

    return vids_win

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

# @threadsafe_generator
# def generate_training_windows(video_data, batch_size = 32):
#     '''
#     Generates training windows batchwise
#     Assumes vids_dict is h5py structure
#     '''
#     print("outside while=true")
#     while True:
#         training_sample_count = video_data["Windows"].shape[0]
#         windows = np.empty((0,) + video_data["Windows"].shape[1:]) # shape: 0 + (8, 128, 128, 1)
#
#         training_sample_idxs = range(0, training_sample_count)
#         vid_chunk_size = 1000
#         chunk_index = 0
#
#         chunks = int(training_sample_count / vid_chunk_size) # total chunks
#         remainder_chunks = training_sample_count % vid_chunk_size # rest windows
#         if remainder_chunks:
#             chunks = chunks + 1
#         print("total chunks", chunks)
#
#         batches = int(training_sample_count / batch_size) # total batches
#         remainder_samples = training_sample_count % batch_size # rest windows
#         if remainder_samples:
#             batches = batches + 1
#         print("total batches", batches)
#
#         for idx in range(0, batches):
#             start_time = time.time()
#             print("batch", idx+1)
#
#             if (len(windows) < batch_size and chunk_index < chunks):
#
#                 if chunk_index == chunks - 1:
#                     chunk_idxs = training_sample_idxs[chunk_index * vid_chunk_size:]
#                 else:
#                     chunk_idxs = training_sample_idxs[chunk_index * vid_chunk_size : chunk_index * vid_chunk_size + vid_chunk_size]
#
#                 windows = np.append(windows, video_data["Windows"][chunk_idxs], axis=0)
#                 chunk_index += 1
#
#             if (len(windows) < batch_size):
#                 batch_data, windows = windows, np.empty((0,) + video_data["Windows"].shape[1:])
#             else:
#                 batch_data, windows = windows[:batch_size], windows[batch_size:]
#
#             yield (batch_data, batch_data)
#             print("Batch Time %.6f s or %.5f mins" % (time.time() - start_time, (time.time() - start_time) / 60))

@threadsafe_generator
def generate_training_windows(video_data, batch_size = 32):
    '''
    Generates training windows batchwise
    Assumes vids_dict is h5py structure
    '''
    print("outside while=true")
    while True:
        training_sample_count = video_data["Windows"].shape[0]
        training_sample_idxs = range(0, training_sample_count)
        batches = int(training_sample_count / batch_size) # total batches
        remainder_samples = training_sample_count % batch_size # rest windows
        if remainder_samples:
            batches = batches + 1
        print("total batches", batches)

        for idx in range(0, batches):
            start_time = time.time()
            print("batch", idx+1)
            if idx == batches - 1:
                batch_idxs = training_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = training_sample_idxs[idx*batch_size:idx*batch_size+batch_size]

            yield (video_data["Windows"][batch_idxs], video_data["Windows"][batch_idxs])
            print("Batch Time %.6f s or %.5f mins" % (time.time() - start_time, (time.time() - start_time) / 60))

# @threadsafe_generator
# def generate_training_windows(vids_dict, stride, win_len, img_width, img_height, batch_size=32):
#     '''
#     returns windows batch wise
#     Assumes vids_dict is h5py structure, ie. vids_dict = hf['Data_2017/UR/Raw/Split_by_video']
#     data set must contain atleast win_len frames
#     '''
#     while True:
#         vid_list = [len(vid['Data'][:]) for vid in list(vids_dict.values())]
#
#         training_sample_count = sum([int(np.floor((val - win_len) / stride)) + 1 for val in vid_list])  # total windows
#         batches = int(training_sample_count / batch_size)  # total batches
#         remainder_samples = training_sample_count % batch_size  # rest windows
#         if remainder_samples:
#             batches = batches + 1
#
#         print("total batches", batches)
#         windows = np.empty((0, win_len, img_width, img_height, 1))
#         num_vids = len(vid_list)
#         vid_names = list(vids_dict.keys())
#         vids = list(vids_dict.values())
#         vid_index = 0
#
#         for idx in range(0, batches):
#             start_time = time.time()
#             if (len(windows) < 5000 and vid_index < num_vids):
#                 print('windowing vid at', vid_names[vid_index])
#                 vid = vids[vid_index]['Data'][:]
#                 vid = vid.reshape(len(vid), img_width, img_height, 1)  # bcoz orig: height, width
#                 vid_windowed = create_windowed_arr(vid, stride, win_len)
#                 print('windowed vid shape', vid_windowed.shape)
#                 windows = np.append(windows, vid_windowed, axis=0)
#                 vid_index += 1
#
#             if (len(windows) < batch_size):
#                 batch_data, windows = windows, np.empty((0, win_len, img_width, img_height, 1))
#             else:
#                 batch_data, windows = windows[:batch_size], windows[batch_size:]
#
#             yield (batch_data, batch_data)
#             print("Batch %d Time %.6f s or %.5f mins" % (idx+1, time.time() - start_time, (time.time() - start_time) / 60))

def create_windowed_arr_per_vid(vids_dict, stride, win_len, img_width, img_height):

    '''
    returns windows made of all videos
    Assumes vids_dict is h5py structure
    data set must contain atleast win_len frames
    '''

    vid_list = [len(vid['Data'][:]) for vid in list(vids_dict.values())]
    num_windowed = sum([int( np.floor((val-win_len)/stride) ) + 1 for val in vid_list])
    output_shape = (num_windowed, win_len, img_width, img_height, 1)
    print(output_shape)
    total = np.zeros(output_shape)
    i = 0
    for vid, name in zip(vids_dict.values(), vids_dict.keys()):
        print('windowing vid at', name)
        vid = vid['Data'][:]
        vid = vid.reshape(len(vid), img_width, img_height, 1) # bcoz orig: height, width
        vid_windowed = create_windowed_arr(vid, stride, win_len)
        print('windowed vid shape', vid_windowed.shape)
        total[i : i+len(vid_windowed)] = vid_windowed
        i += len(vid_windowed)

    return total


def create_windowed_arr(arr, stride, win_len):

    """vids_dict['ADL1'].file
    Creates windows from frames of 1 video!
    arr: array of imgs
    """

    img_width, img_height = arr.shape[1], arr.shape[2]

    output_length = int(np.floor((len(arr) - win_len) / stride)) + 1
    output_shape = (output_length, win_len, img_width, img_height, 1)
    total = np.zeros(output_shape)
    i = 0
    while i < output_length:
        next_chunk = np.array([arr[i+j] for j in range(win_len)])
        # Can use np.arange if want to use time step i.e. np.arange(0,win_len,dt)
        total[i] = next_chunk
        i = i + stride

    return total



def load_data(split_by_vid_or_class = 'Split_by_vid', raw = False, img_width = 64, \
    img_height = 64, vid_class = 'NonFall', dset = 'Thermal_Dummy'):

    """
    Note :to use this function, need to have downloaded h5py for dset,
    and placed in ./H5Data directory, or have downloaded data set,
    extracted frames, and placed them in directory structure specified in h5py_init.py

    Loads data from h5py file, returns a dictionary, the properties of which depend on
    params vid_class and split_by_vid_or_class

    Params:
    	str split_by_vid_or_class: must be one of "Split_by_vid" or "Split_by_class".
    	If "Split_by_vid", the returned dictionary will have key-value pairs for each video.
    	Otherwise, will have key-value pairs for data and labels
    	bool raw: if true, data will be not processed (mean centering and intensity scaling)
    	int img_wdith: width of images
    	int img_height: height of images
        str dset: dataset to be loaded
    	str vid_class: must be one of "NonFall" or "Fall".
    	if split_by_vid_or_class is "Split_by_class", will load only class given by vid_class

    Returns:
    	h5py group data_dict: returns h5py nested group containing strucutred view of data. With

					Split_by_class
						NonFall
							Data
								<HDF5 dataset "Data": shape (samples, img_height*img_width), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (samples,), type "<i4">

					Split_by_video
						ADL1
							Data
								<HDF5 dataset "Data": shape (1397, 4096), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (1397,), type "<i4">
						ADL2
							Data
								<HDF5 dataset "Data": shape (3203, 4096), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (3203,), type "<i4">

							.
							.
							.
						Fall1
							Data
								<HDF5 dataset "Data": shape (49, 4096), type "<f8">
							Labels
								<HDF5 dataset "Labels": shape (49,), type "<i4">
                            .
                            .
                            .


        See h5py_init documentation for more details on creation of the H5 Data.


    """


    path = './H5Data/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    # Create h5 file if it does not exist already!
    if not os.path.isfile(path):
        print('h5py path {} not found, attempting to create h5 file..'.format(path))
        init_videos(img_width = img_width, img_height = img_height, raw = False, dset = dset)
        init_data_by_class(vid_class = vid_class, dset = dset, raw = False,
                           img_width = img_width, img_height = img_height)

    #else:
    #print('h5py path found, loading data_dict..')
    if split_by_vid_or_class == 'Split_by_class':
        if raw == False:
            root_path = dset + '/Processed/' + split_by_vid_or_class + '/' + vid_class
        else:
            root_path = dset + '/Raw/'+ split_by_vid_or_class + '/' + vid_class
    else:
        if raw == False:
            root_path = dset + '/Processed/' + split_by_vid_or_class
        else:
            root_path = dset + '/Raw/'+ split_by_vid_or_class

    print('getting data at group', root_path)

    with h5py.File(path, 'r') as hf:
        data_dict = hf[root_path]['Data'][:]

    return data_dict



def make_windows_h5_file(vids_dict, stride, win_len, img_width, img_height, dset="Thermal-Intrusion"):
    '''

    Creates or overwrites h5py group corresponding to root_path (in body),
    for the h5py file located at
    'N:/FallDetection/Fall-Data/H5Data/Windows-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)
    For info on h5py: http://docs.h5py.org/en/stable/quick.html#quick

    The h5py group of nested groups is structured as follows:
    Windows
        Data
            <HDF5 dataset "Data": shape (#windows, win_len, width, height, 1), type "<f8">

    '''

    path = root_drive + '/H5Data/Windows-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height)

    if os.path.isfile(path):
        print('Window h5 file already exists!')
    else:
        print('No windows data file exists yet; initializing')

        with h5py.File(path, 'a') as hf:

            #root = hf.create_group(dset)
            #grp = root.create_group('Windows')
            #grp['Data'] = np.empty((0, win_len, img_width, img_height, 1))
            first_vid = True
            for vid, name in zip(vids_dict.values(), vids_dict.keys()):
                print('windowing vid at', name)
                vid = vid['Data'][:]
                vid = vid.reshape(len(vid), img_width, img_height, 1)  # bcoz orig: height, width
                vid_windowed = create_windowed_arr(vid, stride, win_len)
                print('windowed vid shape', vid_windowed.shape)

                if(first_vid):
                    hf.create_dataset('Windows', data=vid_windowed, chunks=True, maxshape=(None, win_len, img_width, img_height, 1))
                    first_vid = False
                else:
                    hf["Windows"].resize((hf["Windows"].shape[0] + vid_windowed.shape[0]), axis=0)
                    hf["Windows"][-vid_windowed.shape[0]:] = vid_windowed

            hf.close()

    return path