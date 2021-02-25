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
import scipy
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

from utils import apply_reverb, read_wav
import corpus
import mir_eval
from pypesq import pypesq

import pyroomacoustics as pra
import roomsimove_single
import olafilt


def load_file(files):
    s1, _  = load(files[0], sr=16000)
    s2, _  = load(files[1], sr=16000)

    # s1, s2 = map(read_wav, files)

    if len(s1) > len(s2):
        pad_length = len(s1) - len(s2)
        s2 = np.pad(s2, (0,pad_length), 'reflect')
    else:
        pad_length = len(s2) - len(s1)
        s1 = np.pad(s1, (0,pad_length), 'reflect')
    return s1, s2

def do_reverb(s1,s2):
    corners = np.array([[0,0], [0,8], [8,8], [8,0]]).T  # [x,y]
    room = pra.Room.from_corners(corners)
    room.extrude(5.)

    room.add_source([8.,4.,1.6], signal=s1)
    # room.add_source([2.,4.,1.6], signal=s2)
    #[[X],[Y],[Z]]
    R = np.asarray([[4.75,5.5],[2.,2.],[1.,1]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    room.simulate()

    return room

def do_stft(s1, s2, room):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Y1 = stft(room.mic_array.signals[0,:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    Y2 = stft(room.mic_array.signals[1,:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    X1 = stft(s1, n_fft=nfft, hop_length=hop, win_length=win)
    X2 = stft(s2, n_fft=nfft, hop_length=hop, win_length=win)

    return Y1, Y2, X1, X2

def do_reverb_oldskool(s1,s2, rt60=0.4):
    room_dim = [8, 8, 5] # in meters
    mic_pos1 = [4.75, 2, 1] # in  meters
    mic_pos2 = [2, 2, 1] # in  meters

    sampling_rate = 16000

    mic_positions = [mic_pos1, mic_pos2]
    rir = roomsimove_single.do_everything(room_dim, mic_positions, [8,4,1.6], rt60)

    data_rev_ch1 = olafilt.olafilt(rir[:,0], s1)
    data_rev_ch2 = olafilt.olafilt(rir[:,1], s1)
    return data_rev_ch1, data_rev_ch2

def do_stft_oldskool(s1, s2, m1, m2):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Y1 = stft(m1[:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    Y2 = stft(m2[:len(s1)], n_fft=nfft, hop_length=hop, win_length=win)
    X1 = stft(s1, n_fft=nfft, hop_length=hop, win_length=win)
    X2 = stft(s2, n_fft=nfft, hop_length=hop, win_length=win)

    return Y1, Y2, X1, X2

def correlation(X1, X2, Y1, Y2):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    Gxx = X1 * np.conj(X1)
    Gxyx = X1 * Y1 * np.conj(X1)
    Gyxy = Y1 * X1 * np.conj(Y1)
    Gxy = X1 *  np.conj(Y1)
    Gyx = Y1 * np.conj(X1)
    Gyy = Y1 * np.conj(Y1)

    recon_y1_H1 = istft(np.multiply(np.divide(Gxy, Gxx),Y1), hop_length=hop, win_length=win) * 1000
    recon_y1_H2 = istft(np.multiply(np.divide(Gyy, Gyx),Y1), hop_length=hop, win_length=win) * 1000

    return recon_y1_H1, recon_y1_H2

def correlation_Hs(X1, X2, Y1, Y2, s_value=1):
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    F,T = X1.shape

    Gxx = X1 * np.conj(X1)
    Gxy = X1 * np.conj(Y1)
    Gyx = Y1 * np.conj(X1)
    Gyy = Y1 * np.conj(Y1)

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

    # smallU = U[0:s_value,:].reshape(-1, 2*F).T
    # smallV = V[0:s_value,:].reshape(-1, 2*T)

    Hs1 = np.matmul(smallU[:F,:],pinv(smallV[:,T:]).T)
    Hs2 = np.matmul(smallU[F:,:],pinv(smallV[:,T:]).T)
    Hs3 = np.matmul(smallU[:F,:],pinv(smallV[:,:T]).T)
    Hs4 = np.matmul(smallU[F:,:],pinv(smallV[:,:T]).T)


    recon_y1_H1 = istft(np.multiply(pinv(Hs1).T,Y1), hop_length=hop, win_length=win) * 1000
    recon_y1_H2 = istft(np.multiply(pinv(Hs2).T,Y1), hop_length=hop, win_length=win) * 1000
    recon_y1_H3 = istft(np.multiply(pinv(Hs3).T,Y1), hop_length=hop, win_length=win) * 1000
    recon_y1_H4 = istft(np.multiply(pinv(Hs4).T,Y1), hop_length=hop, win_length=win) * 1000

    return recon_y1_H1, recon_y1_H2, recon_y1_H3, recon_y1_H4

def difference(s1, y1):
    if len(s1) > len(y1):
        bss = mir_eval.separation.bss_eval_sources(np.vstack((s1[:len(y1)],s1[:len(y1)])), np.vstack((y1,y1)))
        pesq = pypesq(16000, s1[:len(y1)], y1, 'wb')

        s1 = s1[:len(y1)]
    else:
        bss = mir_eval.separation.bss_eval_sources(np.vstack((s1,s1)), np.vstack((y1[:len(s1)],y1[:len(s1)])))
        pesq = pypesq(16000, s1, y1[:len(s1)], 'wb')

        y1 = y1[:len(s1)]

    nsrc = 1
    nsampl = len(s1)
    flen = 512
    reference_source = np.hstack((s1, np.zeros((flen - 1))))
    estimated_source = np.hstack((y1.reshape((-1,)), np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_source, n=n_fft)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)

    G = np.zeros((nsrc * flen, nsrc * flen))

    ssf = sf * np.conj(sf)
    ssf = np.real(scipy.fftpack.ifft(ssf))
    ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])), r=ssf[:flen])
    G = ss

    D = np.zeros(nsrc * flen)

    ssef = sf * np.conj(sef)
    ssef = np.real(scipy.fftpack.ifft(ssef))
    D = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    try:
        C = np.linalg.solve(G, D).reshape(flen, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, order='F')
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)

    sproj += fftconvolve(C, reference_source)[:nsampl + flen - 1]

    e_spat = sproj - reference_source
    # interference
    e_interf = sproj - reference_source - e_spat
    # artifacts
    e_artif = -reference_source - e_spat - e_interf
    e_artif[:nsampl] += estimated_source[:nsampl]

    s_filt = reference_source + e_spat
    sdr = 10 * np.log10(np.sum(reference_source**2)/ np.sum((e_interf + e_spat + e_artif)**2))
    # sir = np.sum(s_filt**2)/ np.sum(e_interf**2)
    snr = 10 * np.log10(np.sum((reference_source + e_interf)**2) / np.sum((e_spat)**2))
    sar = 10 * np.log10(np.sum((s_filt + e_interf)**2)/ np.sum(e_artif**2))

    print("SAR: "+str(bss[2][0]) + ", SAR: " + str(sar) + ", SNR: " + str(snr) + ", SDR: " + str(sdr)  + ", SDR: " + str(bss[0][0]) + ", interf: " + str(np.sum(e_interf**2)) + ", artif: " +str(np.sum((e_artif)**2)) + ", spat: " + str(np.sum((e_spat)**2)) )
    return bss[2][0], bss[0][0], bss[1][0], np.sum(e_interf**2), np.sum((e_artif)**2), pesq



def difference_H(s, H1, H2):
    SAR_h1, SDR_h1, SIR_h1, artif_h1, interf_h1, pesq_h1 = difference(s, H1)
    SAR_h2, SDR_h2, SIR_h2, artif_h2, interf_h2, pesq_h2 = difference(s, H2)

    return SAR_h1, SDR_h1, SIR_h1, SAR_h2, SDR_h2, SIR_h2, artif_h1, artif_h2, interf_h1, interf_h2, pesq_h1, pesq_h2

def difference_Hs(s, H1, H2, H3, H4):
    SAR_h1, SDR_h1, SIR_h1, artif_h1, interf_h1, pesq_h1 = difference(s, H1)
    # SAR_h2, SDR_h2, SIR_h2, artif_h2, interf_h2, pesq_h2 = difference(s, H2)
    # SAR_h3, SDR_h3, SIR_h3, artif_h3, interf_h3, pesq_h3 = difference(s, H3)
    # SAR_h4, SDR_h4, SIR_h4, artif_h4, interf_h4, pesq_h4 = difference(s, H4)

    return SAR_h1, SDR_h1, SIR_h1, artif_h1, interf_h1,pesq_h1


def mic_change(M1, M2, switch_mics=False):
    if switch_mics:
        return M2, M1
    else:
        return M1, M2

def experiment(s1,s2, results, area, mic, switch_mics=False,
               go_oldskool=False,rt60=0.4, hs=False, s_value=1):
    if go_oldskool:
        m1, m2 = do_reverb_oldskool(s1,s2, rt60)
        M1, M2, S1, S2 = do_stft_oldskool(s1,s2,m1, m2)
    else:
        room = do_reverb(s1,s2)
        M1, M2, S1, S2 = do_stft(s1,s2,room)
    if hs:
        M1, M2 = mic_change(M1,M2,switch_mics)
        H1, H2, H3, H4 = correlation_Hs(S1, S2, M1, M2, s_value)

        SAR_h1, SDR_h1, SIR_h1, artif_h1, interf_h1, pesq_h1 = difference_Hs(s1, H1, H2, H3, H4)

        results[area][mic+"_h1"]["SAR"].append(SAR_h1)
        results[area][mic+"_h1"]["SDR"].append(SDR_h1)
        results[area][mic+"_h1"]["SIR"].append(SIR_h1)
        results[area][mic+"_h1"]["artif"].append(artif_h1)
        results[area][mic+"_h1"]["interf"].append(interf_h1)
        results[area][mic+"_h1"]["PESQ"].append(pesq_h1)



    else:
        M1, M2 = mic_change(M1,M2,switch_mics)
        H1, H2= correlation(S1, S2, M1, M2)

        SAR_h1, SDR_h1, SIR_h1, SAR_h2, SDR_h2, SIR_h2, artif_h1, artif_h2, interf_h1, interf_h2,  pesq_h1, pesq_h2 = difference_H(s1, H1, H2)

        results[area][mic+"_h1"]["SAR"].append(SAR_h1)
        results[area][mic+"_h1"]["SDR"].append(SDR_h1)
        results[area][mic+"_h1"]["SIR"].append(SIR_h1)
        results[area][mic+"_h1"]["artif"].append(artif_h1)
        results[area][mic+"_h1"]["interf"].append(interf_h1)
        results[area][mic+"_h1"]["PESQ"].append(pesq_h1)

        results[area][mic+"_h2"]["SAR"].append(SAR_h2)
        results[area][mic+"_h2"]["SDR"].append(SDR_h2)
        results[area][mic+"_h2"]["SIR"].append(SIR_h2)
        results[area][mic+"_h2"]["artif"].append(artif_h2)
        results[area][mic+"_h2"]["interf"].append(interf_h2)
        results[area][mic+"_h2"]["PESQ"].append(pesq_h2)

def create_results():
    results = {}
    results = create_subresults(results, "room", "mic1_h1")
    results = create_subresults(results, "room", "mic1_h2")
    results = create_subresults(results, "room", "mic1_h3")
    results = create_subresults(results, "room", "mic1_h4")
    results = create_subresults(results, "room", "mic2_h1")
    results = create_subresults(results, "room", "mic2_h2")
    results = create_subresults(results, "room", "mic2_h3")
    results = create_subresults(results, "room", "mic2_h4")
    results = create_subresults(results, "0.4", "mic1_h1")
    results = create_subresults(results, "0.4", "mic1_h2")
    results = create_subresults(results, "0.4", "mic1_h3")
    results = create_subresults(results, "0.4", "mic1_h4")
    results = create_subresults(results, "0.4", "mic2_h1")
    results = create_subresults(results, "0.4", "mic2_h2")
    results = create_subresults(results, "0.4", "mic2_h3")
    results = create_subresults(results, "0.4", "mic2_h4")
    results = create_subresults(results, "1.0", "mic1_h1")
    results = create_subresults(results, "1.0", "mic1_h2")
    results = create_subresults(results, "1.0", "mic1_h3")
    results = create_subresults(results, "1.0", "mic1_h4")
    results = create_subresults(results, "1.0", "mic2_h1")
    results = create_subresults(results, "1.0", "mic2_h2")
    results = create_subresults(results, "1.0", "mic2_h3")
    results = create_subresults(results, "1.0", "mic2_h4")
    results = create_subresults(results, "1.5", "mic1_h1")
    results = create_subresults(results, "1.5", "mic1_h2")
    results = create_subresults(results, "1.5", "mic1_h3")
    results = create_subresults(results, "1.5", "mic1_h4")
    results = create_subresults(results, "1.5", "mic2_h1")
    results = create_subresults(results, "1.5", "mic2_h2")
    results = create_subresults(results, "1.5", "mic2_h3")
    results = create_subresults(results, "1.5", "mic2_h4")


    return results

def create_subresults(results, area, mic):
    if not area in results.keys():
        results[area] = {}
    results[area][mic] = {}
    results[area][mic]["SAR"] = []
    results[area][mic]["SDR"] = []
    results[area][mic]["SIR"] = []
    results[area][mic]["artif"] = []
    results[area][mic]["interf"] = []
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
            print(np.sum(np.array(results[key][subkey]["artif"]))/no_files)
            print(np.sum(np.array(results[key][subkey]["interf"]))/no_files)
            print(np.sum(np.array(results[key][subkey]["PESQ"]))/no_files)

def main():
    results = create_results()
    print("95")
    with open("files_v2.csv") as f:
        lines = f.readlines()

        s_value = 0.95
        no_files = 10
        for file_nr in range(0,no_files):
            files = []
            s1 = lines[file_nr]
            s2 = lines[file_nr+1]

            s1 = s1[:-1]
            s2 = s2[:-1]
            files.append(s1)
            files.append(s2)
            s1,s2 = load_file(files)

            experiment(s1,s2, results, "room", "mic1", hs=True, s_value=s_value)

        print_results(results,no_files)

if __name__ == '__main__':
    main()
