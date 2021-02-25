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
    s1,_ = load(files[0], sr=16000)

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

def correlation(X1, Y1):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Gxx = np.matmul(X1, np.conj(X1.T))
    # Gxyx = X1 * Y1 * np.conj(X1.T)
    # Gyxy = Y1 * X1 * np.conj(Y1.T)
    Gxy = np.matmul(X1,  np.conj(Y1.T))
    Gyx = np.matmul(Y1, np.conj(X1.T))
    Gyy = np.matmul(Y1, np.conj(Y1.T))

    recon_y1_H1 = istft(np.multiply(np.matmul(np.multiply(Gxy, pinv(Gxx)),Y1),Y1), hop_length=hop, win_length=win) * 1000
    recon_y1_H2 = istft(np.multiply(np.matmul(np.multiply(Gyy, pinv(Gyx)),Y1),Y1), hop_length=hop, win_length=win) * 1000

    return recon_y1_H1, recon_y1_H2

def correlation_Hs(X1, Y1, s_value=1):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    F,T = X1.shape

    Gxx = np.matmul(X1, np.conj(X1.T))
    Gxy = np.matmul(X1, np.conj(Y1.T))
    Gyx = np.matmul(Y1, np.conj(X1.T))
    Gyy = np.matmul(Y1, np.conj(Y1.T))

    temp = np.asarray([[Gxx, Gxy],[Gyx, Gyy]]).reshape(2*F,2*F)

    U, s, V = svd(temp)

    tmpsum = 0
    summed = []
    for i in range(len(s)):
        tmpsum += s[i]/sum(s)
        summed.append(tmpsum)
    summed = np.asarray(summed)
    val_percent = np.where(summed>s_value)[0][0]

    smallU = (1.0/ 31.0) * U[:F,:val_percent]
    
    try:
        smallV = 65 * pinv(V[:F,:val_percent])
        Hs1 = np.matmul(smallU,smallV)
        recon_y1_H1 = istft(np.multiply(np.matmul(Hs1,Y1),Y1), hop_length=hop, win_length=win) * 1000

        return recon_y1_H1, True
    except Exception as e:
        print("")
    return None, False

def SISDR(reference_source, estimated_source):
    ref_energy = np.sum((reference_source ** 2), axis=-1, keepdims=True)
    optimal_scaling = np.sum(reference_source * estimated_source, axis=-1, keepdims=True) / ref_energy

    projection = optimal_scaling * reference_source

    noise = estimated_source - projection

    sisdr = 10*np.log10(np.sum(projection ** 2, axis=-1) / np.sum(noise**2,axis=-1))

    return sisdr

def difference(s1, y1):
    if len(s1) > len(y1):
        bss = mir_eval.separation.bss_eval_sources(np.vstack((s1[:len(y1)],s1[:len(y1)])), np.vstack((y1,y1)))
        pesq = pypesq(16000,s1[:len(y1)], y1, 'wb')
        snr = 10 * np.log10(np.sum((s1[:len(y1)]**2)/((s1[:len(y1)] - y1)**2)))
        sisdr = SISDR(s1[:len(y1)], y1)
    else:
        bss = mir_eval.separation.bss_eval_sources(np.vstack((s1,s1)), np.vstack((y1[:len(s1)],y1[:len(s1)])))
        pesq = pypesq(16000, s1, y1[:len(s1)], 'wb')
        snr = 10 * np.log10(np.sum((s1**2)/((s1 - y1[:len(s1)])**2)))
        sisdr = SISDR(s1, y1[:len(s1)])
    return bss[2][0], bss[0][0], bss[1][0], pesq, snr, sisdr

def difference_H(s, H1, H2):
    SAR_h1, SDR_h1, SIR_h1, pesq_h1, snr_h1, sisdr_h1 = difference(s, H1)
    SAR_h2, SDR_h2, SIR_h2, pesq_h2, snr_h2, sisdr_h2 = difference(s, H2)
    print(str(SAR_h1)+","+str( SDR_h1)+","+str( SIR_h1)+","+str( SAR_h2)+","+str( SDR_h2)+","+str( SIR_h2)+","+str( pesq_h1)+","+str( pesq_h2)+","+str( snr_h1)+","+str( snr_h2)+","+str( sisdr_h1)+","+str( sisdr_h2))
    return SAR_h1, SDR_h1, SIR_h1, SAR_h2, SDR_h2, SIR_h2, pesq_h1, pesq_h2, snr_h1, snr_h2, sisdr_h1, sisdr_h2

def difference_Hs(s, H1):
    SAR_h1, SDR_h1, SIR_h1, pesq_h1, snr_h1, sisdr_h1 = difference(s, H1)
    print(str(SAR_h1)+","+str( SDR_h1)+","+str( SIR_h1)+","+str( pesq_h1)+","+str( snr_h1)+","+str( sisdr_h1))
    return SAR_h1, SDR_h1, SIR_h1, pesq_h1, snr_h1, sisdr_h1

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
    results[area][mic]["SNR"] = []
    results[area][mic]["SISDR"] = []

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
            try:
                print(np.sum(np.array(results[key][subkey]["SNR"]))/no_files)
            except e:
                print("SNR error")
            try:
                print(np.sum(np.array(results[key][subkey]["SISDR"]))/no_files)
            except e:
                print("SISDR error")

def calc_reverb(s1, rt60=1, go_oldskool=False):
    if go_oldskool:
        m1, m2 = do_reverb_oldskool(s1, rt60)
        M1, M2, S1 = do_stft_oldskool(s1,m1, m2)
    else:
        room = do_reverb(s1)
        M1, M2, S1 = do_stft(s1,room)
    return M1, M2, S1

def experiment(s1, S1, M1, results,area,  mic,switch_mics=False,
               go_oldskool=False,rt60=0.4, hs=False, s_value=1):
    if hs:
        H1, success = correlation_Hs(S1, M1, s_value)
        if success:
            SAR_h1, SDR_h1, SIR_h1, pesq_h1, snr_h1, sisdr_h1 = difference_Hs(s1, H1)

            results[area][mic+"_h1"]["SAR"].append(SAR_h1)
            results[area][mic+"_h1"]["SDR"].append(SDR_h1)
            results[area][mic+"_h1"]["SIR"].append(SIR_h1)
            results[area][mic+"_h1"]["PESQ"].append(pesq_h1)
            results[area][mic+"_h1"]["SNR"].append(snr_h1)
            results[area][mic+"_h1"]["SNR"].append(sisdr_h1)
    else:
        H1, H2= correlation(S1, M1)

        SAR_h1, SDR_h1, SIR_h1, SAR_h2, SDR_h2, SIR_h2, pesq_h1, pesq_h2, snr_h1, snr_h2, sisdr_h1, sisdr_h2 = difference_H(s1, H1, H2)

        results[area][mic+"_h1"]["SAR"].append(SAR_h1)
        results[area][mic+"_h1"]["SDR"].append(SDR_h1)
        results[area][mic+"_h1"]["SIR"].append(SIR_h1)
        results[area][mic+"_h1"]["PESQ"].append(pesq_h1)
        results[area][mic+"_h1"]["SNR"].append(snr_h1)
        results[area][mic+"_h1"]["SNR"].append(sisdr_h1)

        results[area][mic+"_h2"]["SAR"].append(SAR_h2)
        results[area][mic+"_h2"]["SDR"].append(SDR_h2)
        results[area][mic+"_h2"]["SIR"].append(SIR_h2)
        results[area][mic+"_h2"]["PESQ"].append(pesq_h2)
        results[area][mic+"_h2"]["SNR"].append(snr_h2)
        results[area][mic+"_h1"]["SNR"].append(sisdr_h2)


def main():

    results_h1 = create_results()
    results_95 = create_results()
    results_9999 = create_results()
    results_999999 = create_results()
    no_files = 100
    for file_nr in range(0,no_files):


        s1 = load_file(corpus.experiment_files_timit_train())
        print("room")
        M1, M2, S1 = calc_reverb(s1)
        s_value = 1
        experiment(s1,M1, M2, results_h1, "room", "mic1", s_value=s_value)
        s_value = 0.95
        experiment(s1,M1, M2, results_95, "room", "mic1", s_value=s_value, hs=True)
        s_value = 0.9999
        experiment(s1,M1, M2, results_9999, "room", "mic1", s_value=s_value, hs=True)
        s_value = 0.999999
        experiment(s1,M1, M2, results_999999, "room", "mic1", s_value=s_value, hs=True)
        print("0.4")
        M1, M2, S1 = calc_reverb(s1, rt60=0.4, go_oldskool=True)
        experiment(s1, M1, M2, results_h1, "0.4", "mic1", go_oldskool=True,
                   s_value=s_value)
        s_value = 0.95
        experiment(s1,M1, M2, results_95, "0.4", "mic1", go_oldskool=True,
                        s_value=s_value, hs=True)
        s_value = 0.9999
        experiment(s1,M1, M2, results_9999, "0.4", "mic1", go_oldskool=True,
                    s_value=s_value, hs=True)
        s_value = 0.999999
        experiment(s1,M1, M2, results_999999, "0.4", "mic1", go_oldskool=True,
                        s_value=s_value, hs=True)
        print("1.0")
        M1, M2, S1 = calc_reverb(s1, rt60=1.0, go_oldskool=True)
        experiment(s1,M1, M2, results_h1, "1.0", "mic1", go_oldskool=True,
                        s_value=s_value, rt60=1.0)
        s_value = 0.95
        experiment(s1,M1, M2, results_95, "1.0", "mic1", go_oldskool=True,
                       s_value=s_value, rt60=1.0, hs=True)
        s_value = 0.9999
        experiment(s1,M1, M2, results_9999, "1.0", "mic1", go_oldskool=True,
                        s_value=s_value, rt60=1.0, hs=True)
        s_value = 0.999999
        experiment(s1,M1, M2, results_999999, "1.0", "mic1", go_oldskool=True,
                        s_value=s_value, rt60=1.0, hs=True)
        print("1.5")
        M1, M2, S1 = calc_reverb(s1, rt60=1.5, go_oldskool=True)
        experiment(s1,M1, M2, results_h1, "1.5", "mic1", go_oldskool=True,
                       s_value=s_value, rt60=1.5)
        s_value = 0.95
        experiment(s1,M1, M2, results_95, "1.5", "mic1", go_oldskool=True,
                       s_value=s_value, rt60=1.5, hs=True)
        s_value = 0.9999
        experiment(s1,M1, M2, results_9999, "1.5", "mic1", go_oldskool=True,
                       s_value=s_value, rt60=1.5, hs=True)
        s_value = 0.999999
        experiment(s1,M1, M2, results_999999, "1.5", "mic1", go_oldskool=True,
                       s_value=s_value, rt60=1.5, hs=True)



    print("H1 & H2")
    print_results(results_h1,no_files)
    print("95")
    print_results(results_95,no_files)
    print("9999")
    print_results(results_9999,no_files)
    print("999999")
    print_results(results_999999,no_files)

if __name__ == '__main__':
    main()
