import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

from librosa.core import load
from librosa.core import stft
from librosa.core import istft
from librosa import amplitude_to_db, db_to_amplitude
from librosa.display import specshow
from librosa.output import write_wav

from scipy.signal import butter, lfilter, csd
from scipy.linalg import svd, pinv

from utils import apply_reverb, read_wav
import corpus
import mir_eval
from pypesq import pypesq

import pyroomacoustics as pra
import roomsimove_single
import olafilt

def load_file(files):
    s1, _  = load(files[0], sr=16000)

    return s1

def do_reverb(s1):
    corners = np.array([[0,0], [0,8], [8,8], [8,0]]).T  # [x,y]
    room = pra.Room.from_corners(corners)
    room.extrude(5.)

    room.add_source([8.,4.,1.6], signal=s1)
    #[[X],[Y],[Z]]
    R = np.asarray([[4.75,5.5],[2.,2.],[1.,1]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    room.simulate()

    return room

def do_stft(s1, room):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Y1 = stft(room.mic_array.signals[0,:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    Y2 = stft(room.mic_array.signals[1,:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    X1 = stft(s1, n_fft=nfft, hop_length=hop, win_length=win)

    return Y1, Y2, X1

def do_reverb_oldskool(s1,rt60=0.4):
    room_dim = [8, 8, 5] # in meters
    mic_pos1 = [4.75, 2, 1] # in  meters
    mic_pos2 = [2, 2, 1] # in  meters

    sampling_rate = 16000

    mic_positions = [mic_pos1, mic_pos2]
    rir = roomsimove_single.do_everything(room_dim, mic_positions, [8,4,1.6], rt60)

    data_rev_ch1 = olafilt.olafilt(rir[:,0], s1)
    data_rev_ch2 = olafilt.olafilt(rir[:,1], s1)
    return data_rev_ch1, data_rev_ch2

def do_stft_oldskool(s1, m1, m2):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Y1 = stft(m1[:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    Y2 = stft(m2[:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    X1 = stft(s1, n_fft=nfft, hop_length=hop, win_length=win)

    return Y1, Y2, X1

def correlation(Y1, Y2):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Gxx = Y1 * np.conj(Y1)
    Gxy = Y1 *  np.conj(Y2)
    Gyx = Y2 * np.conj(Y1)
    Gyy = Y2 * np.conj(Y2)

    recon_y1_H1 = istft(np.multiply(np.divide(Gxy, Gxx),Y1), hop_length=hop, win_length=win) * 1000
    recon_y1_H2 = istft(np.multiply(np.divide(Gyy, Gyx),Y1), hop_length=hop, win_length=win) * 1000

    return recon_y1_H1, recon_y1_H2

def correlation_Hs(Y1, Y2, s_value=1):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    F,T = Y1.shape

    Gxx = Y1 * np.conj(Y1)
    Gxy = Y1 * np.conj(Y2)
    Gyx = Y2 * np.conj(Y1)
    Gyy = Y2 * np.conj(Y2)

    temp = np.asarray([[Gxx, Gxy],[Gyx, Gyy]]).reshape(2*F,2*T)

    U, s, V = svd(temp)

    tmpsum = 0
    summed = []
    for i in range(len(s)):
        tmpsum += s[i]/sum(s)
        summed.append(tmpsum)
    summed = np.asarray(summed)
    val_percent = np.where(summed>s_value)[0][0]

    smallU = U[:,:val_percent].reshape(-1, 2*F).T
    smallV = V[:val_percent,:].reshape(-1, 2*T)

    Hs1 = np.matmul(smallU[:F,:],pinv(smallV[:,T:]).T)


    recon_y1_H1 = istft(np.multiply(pinv(Hs1).T,Y1), hop_length=hop, win_length=win) * 1000

    return recon_y1_H1

def difference(s1, y1):
    if len(s1) > len(y1):
        bss = mir_eval.separation.bss_eval_sources(np.vstack((s1[:len(y1)],s1[:len(y1)])), np.vstack((y1,y1)))
        pesq = pypesq(16000, s1[:len(y1)], y1, 'wb')
    else:
        bss = mir_eval.separation.bss_eval_sources(np.vstack((s1,s1)), np.vstack((y1[:len(s1)],y1[:len(s1)])))
        pesq = pypesq(16000, s1, y1[:len(s1)], 'wb')
    return bss[2][0], bss[0][0], bss[1][0], pesq

def difference_H(s, H1, H2):
    SAR_h1, SDR_h1, SIR_h1, pesq_h1 = difference(s, H1)
    SAR_h2, SDR_h2, SIR_h2, pesq_h2 = difference(s, H2)

    return SAR_h1, SDR_h1, SIR_h1, SAR_h2, SDR_h2, SIR_h2, pesq_h1, pesq_h2

def difference_Hs(s, H1):
    SAR_h1, SDR_h1, SIR_h1, pesq_h1 = difference(s, H1)

    return SAR_h1, SDR_h1, SIR_h1, pesq_h1

def create_results():
    results = {}
    results = create_subresults(results, "room", "mic1_h1")
    results = create_subresults(results, "room", "mic1_h2")
    results = create_subresults(results, "0.4", "mic1_h1")
    results = create_subresults(results, "0.4", "mic1_h2")
    results = create_subresults(results, "1.0", "mic1_h1")
    results = create_subresults(results, "1.0", "mic1_h2")
    results = create_subresults(results, "1.5", "mic1_h1")
    results = create_subresults(results, "1.5", "mic1_h2")

    return results

def create_subresults(results, area, mic):
    if not area in results.keys():
        results[area] = {}
    results[area][mic] = {}
    results[area][mic]["SAR"] = []
    results[area][mic]["SDR"] = []
    results[area][mic]["SIR"] = []
    results[area][mic]["PESQ"] = []
    return results

def print_results(results, no_files):
    for key in results.keys():
        print("|--------------"+key+"---------------|")
        for subkey in results[key].keys():
            print("|--------------"+subkey+"---------------|")
            print(np.sum(np.array(results[key][subkey]["SAR"]))/no_files)
            print(np.sum(np.array(results[key][subkey]["SDR"]))/no_files)
            print(np.sum(np.array(results[key][subkey]["SIR"]))/no_files)
            print(np.sum(np.array(results[key][subkey]["PESQ"]))/no_files)

def calc_reverb(s1, rt60=1, go_oldskool=False):
    if go_oldskool:
        m1, m2 = do_reverb_oldskool(s1,rt60)
        M1, M2, S1 = do_stft_oldskool(s1,m1, m2)
    else:
        room = do_reverb(s1)
        M1, M2, S1 = do_stft(s1,room)
    return M1, M2, S1

def experiment(s1, S1, M1, results,area,  mic,switch_mics=False,
               go_oldskool=False,rt60=0.4, hs=False, s_value=1):
    if hs:
        H1 = correlation_Hs(S1, M1, s_value)

        SAR_h1, SDR_h1, SIR_h1, pesq_h1 = difference_Hs(s1, H1)

        results[area][mic+"_h1"]["SAR"].append(SAR_h1)
        results[area][mic+"_h1"]["SDR"].append(SDR_h1)
        results[area][mic+"_h1"]["SIR"].append(SIR_h1)
        results[area][mic+"_h1"]["PESQ"].append(pesq_h1)

    else:
        H1, H2= correlation(S1, M1)

        SAR_h1, SDR_h1, SIR_h1, SAR_h2, SDR_h2, SIR_h2, pesq_h1, pesq_h2 = difference_H(s1, H1, H2)
        print(str(SAR_h1) + "," + str(SDR_h1) + "," + str(SIR_h1) + "," + str(pesq_h1) + "," + str(SAR_h2) + "," + str(SDR_h2) + "," + str(SIR_h2) + "," + str(pesq_h2))
        results[area][mic+"_h1"]["SAR"].append(SAR_h1)
        results[area][mic+"_h1"]["SDR"].append(SDR_h1)
        results[area][mic+"_h1"]["SIR"].append(SIR_h1)
        results[area][mic+"_h1"]["PESQ"].append(pesq_h1)

        results[area][mic+"_h2"]["SAR"].append(SAR_h2)
        results[area][mic+"_h2"]["SDR"].append(SDR_h2)
        results[area][mic+"_h2"]["SIR"].append(SIR_h2)
        results[area][mic+"_h2"]["PESQ"].append(pesq_h2)


def main():

    results_h1 = create_results()
    with open("files_v2.csv") as f:
        lines = f.readlines()
        no_files = 100
        for file_nr in range(0,len(lines)):
            files = []
            s1 = lines[file_nr]

            s1 = s1[:-1]

            files.append(s1)

            s1 = load_file(files)

            M1, M2, S1 = calc_reverb(s1)
            s_value = 1

            print(lines[file_nr])
            experiment(s1,M1, M2, results_h1, "room", "mic1", s_value=s_value)


        print_results(results_h1,len(lines))

if __name__ == '__main__':
    main()
