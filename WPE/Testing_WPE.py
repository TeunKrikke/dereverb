#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
from numpy.lib import stride_tricks
from scipy.optimize import linprog
# import cvxpy as cp

from librosa import stft as rosa_stft, istft as rosa_istft, amplitude_to_db
from librosa.output import write_wav
from librosa.display import specshow

from wpe import Configrations, WpeMethod
from stft import stft as wpe_stft, istft as wpe_istft

import matplotlib.pyplot as plt

from utils_reverb import load_files

import mir_eval
from pypesq import pesq

import corpus

import pyroomacoustics as pra
import roomsimove_single
import olafilt

from nara_wpe import wpe_v8, wpe_v7, wpe_v6, wpe_v0
from teun_wpe import wpe_v12, wpe_v13, wpe_v14

import time


# In[27]:


def fdndlp(freq_data, d=3, p=15, channels=2, out_channels=2, epochs=10):
    """Frequency-domain variance-normalized delayed liner prediction

    This is the core part of the WPE method. The variance-normalized
    linear prediciton algorithm is implemented in each frequency bin
    separately. Both the input and output signals are in time-domain.

    Args:
        data: A 2-dimension numpy array with shape=(chanels, samples)

    Returns:
        A 2-dimension numpy array with shape=(output_channels, samples)
    """
    freq_num = freq_data.shape[-1]
    drv_freq_data = freq_data[0:out_channels].copy()
    for i in range(freq_num):
        xt = freq_data[:,:,i].T
        drv_xt = ndlp(xt, d, p, channels, out_channels, epochs)
        drv_freq_data[:,:,i] = drv_xt.T
    return drv_freq_data

def ndlp(xt, d=2, p=30, channels=2, out_channels=1, epochs=2):
    """Variance-normalized delayed liner prediction

    Here is the specific WPE algorithm implementation. The input should be
    the reverberant time-frequency signal in a single frequency bin and
    the output will be the dereverberated signal in the corresponding
    frequency bin.

    Args:
        xk: A 2-dimension numpy array with shape=(frames, input_chanels)

    Returns:
        A 2-dimension numpy array with shape=(frames, output_channels)
    """
    cols = xt.shape[0] - d
    xt_buf = xt[:,0:out_channels]
    xt = np.concatenate(
        (np.zeros((p - 1, channels)), xt),
        axis=0)
    xt_tmp = xt[:,::-1].copy()
    frames = stride_tricks.as_strided(
        xt_tmp,
        shape=(channels * p, cols),
        strides=(xt_tmp.strides[-1], xt_tmp.strides[-1] * channels)) # past signal
    frames = frames[::-1]
    sigma2 = np.mean(1 / (np.abs(xt_buf[d:]) ** 2), axis=1) # time varying variance
    dt = xt_buf[d:]
    for _ in range(epochs):
        x_cor_m = np.dot(
                np.dot(frames, np.diag(sigma2)),
                np.conj(frames.T))
        x_cor_v = np.dot(
            frames,
            np.conj(xt_buf[d:] * sigma2.reshape(-1, 1))) # calculating phi
        coeffs = np.dot(np.linalg.inv(x_cor_m), x_cor_v) # updating c where x_cor_m is PHI and x_cor_v is phi

        dt = xt_buf[d:] - np.dot(frames.T, np.conj(coeffs)) # desired signal at time t
        sigma2 = np.mean(1 / (np.abs(dt) ** 2), axis=1) # time varying variance

    return np.concatenate((xt_buf[0:d], dt))


# In[28]:

#
# def laplacian_start(freq_data, d=3, p=15, channels=2, out_channels=2, epochs=10):
#     """Frequency-domain variance-normalized delayed liner prediction
#
#     This is the core part of the WPE method. The variance-normalized
#     linear prediciton algorithm is implemented in each frequency bin
#     separately. Both the input and output signals are in time-domain.
#
#     Args:
#         data: A 2-dimension numpy array with shape=(chanels, samples)
#
#     Returns:
#         A 2-dimension numpy array with shape=(output_channels, samples)
#     """
#
#     freq_num = freq_data.shape[-1]
#     drv_freq_data = freq_data[0:out_channels].copy()
#     for k in range(freq_num):
#         xt = freq_data[:,:,k].T
#         drv_xt = laplacian_model(xt, d, p, channels, out_channels, epochs)
#         drv_freq_data[:,:,k] = drv_xt.T
#
#     return drv_freq_data
#
# def laplacian_model(xt, d=3, p=15, channels=2, out_channels=2, epochs=10):
#     """Variance-normalized delayed liner prediction
#
#     Here is the specific WPE algorithm implementation. The input should be
#     the reverberant time-frequency signal in a single frequency bin and
#     the output will be the dereverberated signal in the corresponding
#     frequency bin.
#
#     Args:
#         xk: A 2-dimension numpy array with shape=(frames, input_chanels)
#         d: prediction delay
#         p: regression kernel
#
#     Returns:
#         A 2-dimension numpy array with shape=(frames, output_channels)
#     """
#     lambda_t = ((np.abs(xt[:,0].real) + np.abs(xt[:,0].imag))[:-d] ** 2).reshape(-1,1)
#     cols = xt.shape[0] - d
#     xt_buf = xt[:,0].reshape(-1,1)
#     xt = np.concatenate(
#         (np.zeros((p - 1, channels)), xt),
#         axis=0)
#     xt_tmp = xt[:,::-1].copy()
#     frames = stride_tricks.as_strided(
#         xt_tmp,
#         shape=(channels * p, cols),
#         strides=(xt_tmp.strides[-1], xt_tmp.strides[-1] * channels)) # past signal
#     frames = frames[::-1]
#     size = lambda_t.shape
# #     print(size)
#     for _ in range(epochs):
#         lambda_sqrt= (np.sqrt(lambda_t)/2).reshape(-1)
#         frames_r = np.real(frames.T)
#         frames_s = (np.imag(frames.T)-np.real(frames.T))
#         xt_r = np.real(xt_buf[d:])
#         xt_i = np.imag(xt_buf[d:])
#         t = cp.Variable((2*size[0], 1))
#         g = cp.Variable((channels * p, 1))
#         h = cp.Variable((channels * p, 1))
#
#         opt = cp.Minimize(cp.norm(t, 1))
#         prob = cp.Problem(opt, [cp.abs(xt_r-frames_r@g)<=cp.reshape(cp.multiply(lambda_sqrt,t[size[0]-1:-1,0]), size), cp.abs(np.imag(xt_buf[d:])-frames_s@g)<=cp.reshape(cp.multiply(lambda_sqrt,t[size[0]:,0]), size)])
#
#         optimal_value = prob.solve()
#         g = g.value
#         g_t = g
#
#         dt = xt_buf[d:] - np.dot(g_t.T,frames).T # desired signal at time t
#
#         lambda_t = np.maximum((np.abs(dt.real) + np.abs(dt.imag)) ** 2, np.ones(lambda_t.shape)*0.0000001)
#     return np.concatenate((xt_buf[0:d], dt))


# In[52]:


def cauchy_start(freq_data, d=3, p=15, channels=2, out_channels=1, epochs=10):
    """Frequency-domain variance-normalized delayed liner prediction

    This is the core part of the WPE method. The variance-normalized
    linear prediciton algorithm is implemented in each frequency bin
    separately. Both the input and output signals are in time-domain.

    Args:
        data: A 2-dimension numpy array with shape=(chanels, samples)

    Returns:
        A 2-dimension numpy array with shape=(output_channels, samples)
    """
#     out_num = 2

    freq_num = freq_data.shape[-1]
    drv_freq_data = freq_data[0:out_channels].copy()
    for k in range(freq_num):
        xt = freq_data[:,:,k].T
        drv_xt = cauchy_model(xt, d, p, channels, out_channels, epochs)
        drv_freq_data[:,:,k] = drv_xt.T
    return drv_freq_data

def cauchy_model(xt, d=2, p=30, channels=2, out_channels=1, epochs=2):
    """Variance-normalized delayed liner prediction

    Here is the specific WPE algorithm implementation. The input should be
    the reverberant time-frequency signal in a single frequency bin and
    the output will be the dereverberated signal in the corresponding
    frequency bin.

    Args:
        xk: A 2-dimension numpy array with shape=(frames, input_chanels)

    Returns:
        A 2-dimension numpy array with shape=(frames, output_channels)
    """
    cols = xt.shape[0] - d
    eta = 10**-6
    xt_buf = xt[:,0:out_channels]

    xt = np.concatenate(
        (np.zeros((p - 1, channels)), xt),
        axis=0)
    xt_tmp = xt[:,::-1].copy()
    frames = stride_tricks.as_strided(
        xt_tmp,
        shape=(channels * p, cols),
        strides=(xt_tmp.strides[-1], xt_tmp.strides[-1] * channels)) # past signal
    frames = np.abs(frames[::-1] ** 2)
    dt = xt_buf[d:]
    try:
        for _ in range(epochs):
            lambda_k = (np.abs(dt) ** 2).reshape(-1,)# time varying variance

            lambda_frames = np.dot(np.linalg.inv(np.diag(lambda_k+eta)),
                                   2*np.pi*(frames**2 + lambda_k ** 2).T**1.5).T
            # lambda_frames = np.dot(inv_lstsq(np.diag(lambda_k)),
            #                        2*np.pi*(frames**2 + lambda_k ** 2).T**1.5).T
            x_cor_m = np.dot(
                    lambda_frames,
                    np.conj(lambda_frames.T))

            x_cor_v = np.dot(
                lambda_frames,
                np.conj(xt_buf[d:])) # calculating phi


            g_t = np.dot(np.linalg.inv(x_cor_m+np.eye(x_cor_m.shape[1])*eta), x_cor_v) # updating c where x_cor_m is PHI and x_cor_v is phi
            # g_t = np.dot(inv_lstsq(x_cor_m), x_cor_v) # updating c where x_cor_m is PHI and x_cor_v is phi

            dt = xt_buf[d:] - np.dot(frames.T, np.conj(g_t)) # desired signal at time t
    except np.linalg.LinAlgError as err:
        temp = 0
    return np.concatenate((xt_buf[0:d], dt))

def cauchy_start2(freq_data, d=3, p=15, channels=2, out_channels=1, epochs=10):
    """Frequency-domain variance-normalized delayed liner prediction

    This is the core part of the WPE method. The variance-normalized
    linear prediciton algorithm is implemented in each frequency bin
    separately. Both the input and output signals are in time-domain.

    Args:
        data: A 2-dimension numpy array with shape=(chanels, samples)

    Returns:
        A 2-dimension numpy array with shape=(output_channels, samples)
    """
#     out_num = 2

    freq_num = freq_data.shape[-1]
    drv_freq_data = freq_data[0:out_channels].copy()
    for k in range(freq_num):
        xt = freq_data[:,:,k].T
        drv_xt = cauchy_model2(xt, d, p, channels, out_channels, epochs)
        drv_freq_data[:,:,k] = drv_xt.T
    return drv_freq_data

def cauchy_model2(xt, d=2, p=30, channels=2, out_channels=1, epochs=2):
    """Variance-normalized delayed liner prediction

    Here is the specific WPE algorithm implementation. The input should be
    the reverberant time-frequency signal in a single frequency bin and
    the output will be the dereverberated signal in the corresponding
    frequency bin.

    Args:
        xk: A 2-dimension numpy array with shape=(frames, input_chanels)

    Returns:
        A 2-dimension numpy array with shape=(frames, output_channels)
    """
    cols = xt.shape[0] - d
    xt_buf = xt[:,0:out_channels]
    xt = np.concatenate(
        (np.zeros((p - 1, channels)), xt),
        axis=0)
    xt_tmp = xt[:,::-1].copy()
    frames = stride_tricks.as_strided(
        xt_tmp,
        shape=(channels * p, cols),
        strides=(xt_tmp.strides[-1], xt_tmp.strides[-1] * channels)) # past signal
    frames = frames[::-1]
    dt = xt_buf[d:]
    for _ in range(epochs):
        lambda_k = (np.abs(xt_buf[d:]) ** 2).reshape(-1,)# time varying variance
        lambda_frames = np.dot((np.abs(frames)**2 + lambda_k),
                              np.linalg.inv(np.diag(3*lambda_k)))
        # print(np.sum(lambda_frames))
        x_cor_m = np.dot(
                lambda_frames,
                np.conj(lambda_frames.T))

        x_cor_v = np.dot(
            lambda_frames,
            np.conj(xt_buf[d:])) # calculating phi


        g_t = np.dot(np.linalg.inv(x_cor_m), x_cor_v) # updating c where x_cor_m is PHI and x_cor_v is phi

        dt = xt_buf[d:] - np.dot(frames.T, np.conj(g_t)) # desired signal at time t
    return np.concatenate((xt_buf[0:d], dt))

def cauchy_start3(freq_data, d=3, p=15, channels=2, out_channels=1, epochs=10):
    """Frequency-domain variance-normalized delayed liner prediction

    This is the core part of the WPE method. The variance-normalized
    linear prediciton algorithm is implemented in each frequency bin
    separately. Both the input and output signals are in time-domain.

    Args:
        data: A 2-dimension numpy array with shape=(chanels, samples)

    Returns:
        A 2-dimension numpy array with shape=(output_channels, samples)
    """
#     out_num = 2

    freq_num = freq_data.shape[-1]
    drv_freq_data = freq_data[0:out_channels].copy()
    for k in range(freq_num):
        xt = freq_data[:,:,k].T
        drv_xt = cauchy_model3(xt, d, p, channels, out_channels, epochs)
        drv_freq_data[:,:,k] = drv_xt.T
    return drv_freq_data

def cauchy_model3(xt, d=2, p=30, channels=2, out_channels=1, epochs=2):
    """Variance-normalized delayed liner prediction

    Here is the specific WPE algorithm implementation. The input should be
    the reverberant time-frequency signal in a single frequency bin and
    the output will be the dereverberated signal in the corresponding
    frequency bin.

    Args:
        xk: A 2-dimension numpy array with shape=(frames, input_chanels)

    Returns:
        A 2-dimension numpy array with shape=(frames, output_channels)
    """
    cols = xt.shape[0] - d
    xt_buf = xt[:,0:out_channels]
    xt = np.concatenate(
        (np.zeros((p - 1, channels)), xt),
        axis=0)
    xt_tmp = xt[:,::-1].copy()
    frames = stride_tricks.as_strided(
        xt_tmp,
        shape=(channels * p, cols),
        strides=(xt_tmp.strides[-1], xt_tmp.strides[-1] * channels)) # past signal
    frames = frames[::-1]
    dt = xt_buf[d:]
    for _ in range(epochs):
        lambda_k = (np.abs(dt) ** 2).reshape(-1,)# time varying variance
        lambda_frames = lambda_k/np.dot(np.diag(3*lambda_k),
                              np.linalg.pinv((np.abs(frames)**2 + lambda_k**2))).T

#         lambda_frames = np.dot(3*(np.abs(frames)**2),
#                               np.linalg.inv(np.diag(lambda_k**2)))

        x_cor_m = np.dot(
                lambda_frames,
                np.conj(lambda_frames.T))

        x_cor_v = np.dot(
            lambda_frames,
            np.conj(xt_buf[d:])) # calculating phi


        g_t = np.dot(np.linalg.inv(x_cor_m), x_cor_v) # updating c where x_cor_m is PHI and x_cor_v is phi
        dt = xt_buf[d:] - np.dot(frames.T, np.conj(g_t)) # desired signal at time t
#         dt = np.dot(frames.T, np.conj(g_t))
    return np.concatenate((xt_buf[0:d], dt))

def inv_lstsq(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]

# In[61]:


def do_reverb(s1, locs=[[3.2,1.55,1.6],[4.,6.,1.6]]):
    # corners = np.array([[0,0], [0,8], [8,8], [8,0]]).T  # [x,y]
    corners = np.array([[0,0], [0,5], [4,5], [4,0]]).T  # [x,y]
    room = pra.Room.from_corners(corners)
    # room.extrude(5.)
    room.extrude(3.)

    room.add_source(locs[0], signal=s1)
    #[[X],[Y],[Z]]
    # R = np.asarray([[4.75,5.5],[2.,2.],[1.,1]])
    R = np.asarray([[1.5,1.6],[2.,2.],[1.,1]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    room.simulate()

    return room, locs

def apply_reverb(speaker, source_pos = [3.2, 1.55, 1.6], rt60 = 0.4):
    # room_dim = [5, 4, 3] # in meters
    # mic_pos1 = [2, 1.5, 1] # in  meters
    # mic_pos2 = [2, 1.6, 1] # in  meters

    source_pos = [8, 4, 1.6]
    room_dim = [8, 8, 5] # in meters
    mic_pos1 = [4.75, 2, 1] # in  meters
    mic_pos2 = [2, 2, 1] # in  meters

    sampling_rate = 16000

    mic_positions = [mic_pos1, mic_pos2]
    rir = roomsimove_single.do_everything(room_dim, mic_positions, source_pos, rt60)


    data_rev_ch1 = olafilt.olafilt(rir[:,0], speaker)
    data_rev_ch2 = olafilt.olafilt(rir[:,1], speaker)
    return np.array([data_rev_ch1, data_rev_ch2])

# In[60]:


def SDR_images(s1, y1):
    reference_sources = np.hstack((s1,s1))
    estimated_sources = np.hstack((y1,y1))
    (sdr, isr, sir, sar, perm) = mir_eval.separation.bss_eval_images(reference_sources, estimated_sources)
    return sdr[0]

def SDR_sources(s1, y1):
    reference_sources = np.hstack((s1,s1))
    estimated_sources = np.hstack((y1,y1))
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)
    return sdr[0]

def SISDR(s1, y1):
    ref_energy = np.sum((s1 ** 2), axis=-1, keepdims=True)
    optimal_scaling = np.sum(s1 * y1, axis=-1, keepdims=True) / ref_energy

    projection = optimal_scaling * s1
    noise = y1 - projection

    sisdr = 10*np.log10(np.sum(projection ** 2, axis=-1) / np.sum(noise**2, axis=-1))
    return sisdr


# In[78]:


def calc_score(X1, s1, time, sr=16000):
    x1 = wpe_istft(X1, frame_size=512, overlap=0.5)[0,:]
    x1 = x1 / np.abs(x1).max()
    length = len(x1)
    if len(s1) < length:
        length = len(s1)
    score_x1 = pesq(s1[:length], x1[:length], sr)
    sisdr_x1 = SISDR(s1[:length], x1[:length])
    sdrs_x1 = SDR_images(s1[:length], x1[:length])
    sdri_x1 = SDR_sources(s1[:length], x1[:length])
    print(str(score_x1)+','+str(sisdr_x1)+','+str(sdrs_x1)+','+str(sdri_x1)+','+str(time))


# In[64]:
# windows = ["blackman", "hann", "hamming", "barthann", "bartlett", "triang"]
windows = ["hann"]
for window in windows:
    print(window)
    for rt60 in range(1,11):
        print(rt60)
        for i in range(100):
            s1, s2 = load_files(corpus.experiment_files_timit_train())
            sr=16000

            rt60_nw = rt60 / 10
            y = apply_reverb(s1, rt60=rt60_nw)

            nfft= 2048
            win = 1024
            hop = int(nfft/8)

            Y = wpe_stft(y / np.abs(y).max(), frame_size=512, overlap=0.5, window_name=window)

            nara_Y = Y.transpose(2,0,1)
            # start = time.time()
            # X1 = wpe_v8(nara_Y)
            # timex1 = time.time() - start
            start = time.time()
            X2 = wpe_v7(nara_Y)
            timex2 = time.time() - start
            start = time.time()
            # X3 = wpe_v6(nara_Y)
            # timex3 = time.time() - start
            # start = time.time()
            # X4 = wpe_v0(nara_Y)
            # timex4 = time.time() - start
            # start = time.time()
            # X5 = fdndlp(Y)
            # timex5 = time.time() - start
            # start = time.time()
            # # # X6 = laplacian_start(Y)
            # X7 = cauchy_start(Y)
            # timex7 = time.time() - start
            # start = time.time()
            # X8 = cauchy_start2(Y)
            # timex8 = time.time() - start
            # start = time.time()
            # X9 = cauchy_start3(Y)
            # timex9 = time.time() - start
            # start = time.time()
            #
            X12 = wpe_v12(nara_Y)
            timex12 = time.time() - start
            start = time.time()
            X13 = wpe_v13(nara_Y)
            timex13 = time.time() - start
            start = time.time()
            X14 = wpe_v14(nara_Y)
            timex14 = time.time() - start
            start = time.time()

            # calc_score(X1.transpose(1,2,0), s1, timex1)
            calc_score(X2.transpose(1,2,0), s1, timex2)
            # calc_score(X3.transpose(1,2,0), s1, timex3)
            # calc_score(X4.transpose(1,2,0), s1, timex4)
            # calc_score(X5, s1, timex5)
            # # # calc_score(X6, s1)
            # calc_score(X7, s1, timex7)
            # calc_score(X8, s1, timex8)
            # calc_score(X9, s1, timex9)
            #
            calc_score(X12.transpose(1,2,0), s1, timex12)
            calc_score(X13.transpose(1,2,0), s1, timex13)
            calc_score(X14.transpose(1,2,0), s1, timex14)
