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

class ReverbExperiment(object):
    def __init__(self, window, overlap, fft_bins):
        super(ReverbExperiment, self).__init__()
        self.nfft=fft_bins
        self.win = window
        self.hop = overlap

    def load_file(self, files):
        """
            load two files and make them the same lenght
        """
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

    def do_reverb(self, s1, rt60=0):
        if rt60 is None:
            corners = np.array([[0,0], [0,8], [8,8], [8,0]]).T  # [x,y]
            room = pra.Room.from_corners(corners)
            room.extrude(5.)

            room.add_source([8.,4.,1.6], signal=s1)
            # room.add_source([2.,4.,1.6], signal=s2)
            #[[X],[Y],[Z]]
            R = np.asarray([[4.75,5.5],[2.,2.],[1.,1]])
            room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

            room.simulate()

            return room.mic_array.signals[0,:len(s1)], room.mic_array.signals[1,:len(s1)]
        else:
            room_dim = [8, 8, 5] # in meters
            mic_pos1 = [4.75, 2, 1] # in  meters
            mic_pos2 = [5.5, 2, 1] # in  meters

            sampling_rate = 16000

            mic_positions = [mic_pos1, mic_pos2]
            rir = roomsimove_single.do_everything(room_dim, mic_positions, [8,4,1.6], rt60)

            data_rev_ch1 = olafilt.olafilt(rir[:,0], s1)
            data_rev_ch2 = olafilt.olafilt(rir[:,1], s1)
            return data_rev_ch1[:len(s1)], data_rev_ch2[:len(s1)]

    def do_stft(self, y1, y2):
        Y1 = stft(y1, n_fft=nfft, hop_length=hop, win_length=win)
        Y2 = stft(y2, n_fft=nfft, hop_length=hop, win_length=win)

        return Y1, Y2

    def correlation(self, S1, S2):
        raise NotImplementedError

    def difference(self,x, y):
        """
            Calculate the difference between the two signals
            Output:
                SAR, SDR, SIR, PESQ
        """
        max_signal_length = len(y)
        if len(x) < len(y):
            max_signal_length = len(x)

        gt = np.vstack((x[:max_signal_length],x[:max_signal_length]))
        est = np.vstack((y[:max_signal_length],y[:max_signal_length]))

        bss = mir_eval.separation.bss_eval_sources(gt, est)
        pesq = pypesq(16000, x[:max_signal_length],y[:max_signal_length], 'wb')

        return bss[2][0], bss[0][0], bss[1][0], pesq

    def create_results(self,reverb_times):
        self.results = {}
        for rt60 in reverb_times:
            if rt60 is None:
                rt60 = "room"
            self.results[rt60] = {}
            self.results[rt60]["SAR"] = []
            self.results[rt60]["SDR"] = []
            self.results[rt60]["SIR"] = []
            self.results[rt60]["PESQ"] = []

    def store_result(self,SAR, SDR, SIR, PESQ, rt60):
        if rt60 is None:
            rt60 = "room"
        self.results[rt60]["SAR"].append(SAR)
        self.results[rt60]["SDR"].append(SDR)
        self.results[rt60]["SIR"].append(SIR)
        self.results[rt60]["PESQ"].append(PESQ)

    def print_results(self,no_files):
        for key in self.results.keys():
            print("|--------------"+str(key)+"---------------|")
            print(np.sum(np.array(self.results[key]["SAR"]))/no_files)
            print(np.sum(np.array(self.results[key]["SDR"]))/no_files)
            print(np.sum(np.array(self.results[key]["SIR"]))/no_files)
            print(np.sum(np.array(self.results[key]["PESQ"]))/no_files)


    def run(self, file_list, reverb_times=(None,0.4, 1.0, 1.5), num_files=100):
        self.create_results(reverb_times)
        for file_nr in range(0,num_files, 2):
            s1 = lines[file_nr]
            s2 = lines[file_nr+1]

            s1 = s1[:-1]
            s2 = s2[:-1]

            s1,s2 = self.load_file([s1, s2])

            for rt60 in reverb_times:
                y1,y2 = self.do_reverb(s1,rt60)
                Y1, Y2 = self.do_stft(y1, y2)
                y = self.correlation(Y1, Y2)
                SAR, SDR, SIR, PESQ = self.difference(s1,y)
                self.store_result(SAR, SDR, SIR, PESQ, rt60)

        self.print_results(num_files)



class ReverbExperiment_H1(ReverbExperiment):
    def __init__(self, window, overlap, fft_bins):
        super(ReverbExperiment_H1, self).__init__(window, overlap, fft_bins)

    def correlation(self, Y1, Y2):
        """
            Determine the simple correlation between the reverb and nonreverb signals

        """
        Gxx = Y1 * np.conj(Y1)
        Gxy = Y1 * np.conj(Y2)

        recon_y = istft(np.multiply(np.divide(Gxy, Gxx),Y2),
                            hop_length=self.hop,
                            win_length=self.win) * 1000

        return recon_y

    def __str__(self):
        return 'H1'

class ReverbExperiment_H2(ReverbExperiment):
    def __init__(self, window, overlap, fft_bins):
        super(ReverbExperiment_H2, self).__init__(window, overlap, fft_bins)

    def correlation(self, Y1, Y2):
        """
            Determine the simple correlation between the reverb and nonreverb signals

        """
        Gyx = Y2 * np.conj(Y1)
        Gyy = Y2 * np.conj(Y2)

        recon_y = istft(np.multiply(np.divide(Gyy, Gyx),Y2),
                        hop_length=self.hop, win_length=self.win) * 1000

        return recon_y

    def __str__(self):
        return 'H2'


class ReverbExperiment_Hs(ReverbExperiment):
    def __init__(self, window, overlap, fft_bins, lambda_value=0.95):
        super(ReverbExperiment_Hs, self).__init__(window, overlap, fft_bins)
        self.lambda_value = lambda_value

    def calc_lambda(self, s):
        tmpsum = 0
        summed = []
        for i in range(len(s)):
            tmpsum += s[i]/sum(s)
            summed.append(tmpsum)
        summed = np.asarray(summed)
        return np.where(summed>s_value)[0][0]

    def correlation(self, Y1, Y2):
        F,T = Y1.shape

        Gxx = Y1 * np.conj(Y1)
        Gxy = Y1 * np.conj(Y2)
        Gyx = Y2 * np.conj(Y1)
        Gyy = Y2 * np.conj(Y2)

        Gxyxy = np.asarray([[Gxx, Gxy],[Gyx, Gyy]]).reshape(2*F,2*T)

        U, s, V = svd(Gxyxy)

        val_percentself.calc_lambda(s)

        smallU = U[:,:val_percent].reshape(-1, 2*F).T
        smallV = V[:val_percent,:].reshape(-1, 2*T)

        Hs = np.matmul(smallU[:F,:],pinv(smallV[:,T:]).T)


        recon_y = istft(np.multiply(pinv(Hs).T,Y2),
                            hop_length=self.hop,
                            win_length=self.win) * 1000

        return recon_y
    def __str__(self):
        return 'Hs_'+str(lambda_value*100)

class ReverbMultiExperiment(ReverbExperiment):
    def __init__(self, window, overlap, fft_bins):
        super(ReverbMultiExperiment, self).__init__(window, overlap, fft_bins)

    def create_results(self,methods, reverb_times):
        self.results = {}
        for method in methods:
            self.results[str(method)] = {}
            for rt60 in reverb_times:
                if rt60 is None:
                    rt60 = "room"
                self.results[str(method)][rt60] = {}
                self.results[str(method)][rt60]["SAR"] = []
                self.results[str(method)][rt60]["SDR"] = []
                self.results[str(method)][rt60]["SIR"] = []
                self.results[str(method)][rt60]["PESQ"] = []

    def store_result(self,SAR, SDR, SIR, PESQ, rt60, method):
        if rt60 is None:
            rt60 = "room"
        self.results[method][rt60]["SAR"].append(SAR)
        self.results[method][rt60]["SDR"].append(SDR)
        self.results[method][rt60]["SIR"].append(SIR)
        self.results[method][rt60]["PESQ"].append(PESQ)

    def print_results(self,no_files):
        for key in self.results.keys():
            print("|--------------"+str(key)+"---------------|")
            for subkey in self.results[key].keys():
                print("|--------------"+str(subkey)+"---------------|")
                print(np.sum(np.array(self.results[key][subkey]["SAR"]))/no_files)
                print(np.sum(np.array(self.results[key][subkey]["SDR"]))/no_files)
                print(np.sum(np.array(self.results[key][subkey]["SIR"]))/no_files)
                print(np.sum(np.array(self.results[key][subkey]["PESQ"]))/no_files)

    def run(self,file_list, reverb_times=(None,0.4, 1.0, 1.5), num_files=100,
            methods=None):
        self.create_results(methods,reverb_times)
        for file_nr in range(0,num_files, 2):
            s1 = lines[file_nr]
            s2 = lines[file_nr+1]

            s1 = s1[:-1]
            s2 = s2[:-1]

            s1,s2 = self.load_file([s1, s2])

            for rt60 in reverb_times:
                y1,y2 = self.do_reverb(s1,rt60)
                Y1, Y2 = self.do_stft(y1, y2)
                for method in methods:
                    y = method.correlation(Y1, Y2)
                    SAR, SDR, SIR, PESQ = method.difference(s1,y)
                    self.store_result(SAR, SDR, SIR, PESQ, rt60, str(method))

        self.print_results(num_files)

if __name__ == '__main__':
    nfft=2048
    win = 1024
    hop = int(nfft/8)

    h1 = ReverbExperiment_H1(win, hop, nfft)
    h2 = ReverbExperiment_H2(win, hop, nfft)
    hs = ReverbExperiment_Hs(win, hop, nfft, lambda_value=0.95)
    multi = ReverbMultiExperiment(win, hop, nfft)
    with open("files_v2.csv") as f:
        lines = f.readlines()
        # h1.run(lines, reverb_times=(None, 0.4), num_files=3)
        multi.run(lines, reverb_times=(None, 0.4), num_files=3, methods=[h1, h2])
