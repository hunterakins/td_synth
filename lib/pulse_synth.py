import numpy as np
from matplotlib import pyplot as plt
from env.env.env_loader import Env
from pyat.pyat.env import Box, Beam, Arrival, Arrivals
from pyat.pyat.readwrite import read_arrivals_asc
from ray.arrival_proc import index_to_pos
from scipy.signal import get_window, convolve, hilbert
import os,sys

'''
Description:
Create a useful class for pulse synthesis

Parameters will be pulse shape, frequency resolution, environmental model, NM vs sparc vs ?

Author: Hunter Akins
'''

def create_LFM_pulse(start_freq, end_freq, pulse_width, window='hanning'):
        """ 
        Input:
        start_freq - float 
        Low end of frequency sweep
        end_freq - float
        High end of frequency sweep
        pulse_width - float
        Length of pulse (seconds)
        Output:
        Initialize a TDPulse object and set it as the lone attribute?
        """
        alpha = (end_freq - start_freq) / pulse_width # slope of freq increase
        f_t = lambda t: np.sin(2*np.pi*(start_freq + alpha *t)*t)
        r_t = lambda t: (t < pulse_width) # rectangular window
        s_t = lambda t: r_t(t)*f_t(t)
        dt = 1 /( 8*end_freq)
        """ make pulse length a power of 2 """
        N = np.power(2, int(np.log2(pulse_width/ dt))+1)
        support = np.linspace(0, N*dt, N)
        shape = s_t(support)
        """
        Apply window
        """
        """ find the number of vals in the pulse """
        pulse_N = 0
        while support[pulse_N] < pulse_width:
            pulse_N += 1
        """ get window values """
        win_vals = get_window(window, pulse_N)
        """  apply to shape """
        shape[:pulse_N] = shape[:pulse_N] * win_vals
        shape = hilbert(shape)
        pulse = TDPulse(support, shape,pulse_N)
        return pulse
         
class TDPulse:
    """
    Pulse class. Contains information for synthesis of pulses from a acoustic model...
    """

    def __init__(self, support, shape,pulse_N=None):
        """
        Input:
        support - 1d numpy array
        The time domain values (in seconds) over which the pulse is nonzero
        shape - 1d numpy array dtype =complex128
        pulse_N = int
        number of nonzero values in shape (optional)
        The pulse values on the support
        """ 
        self.support = support 
        self.shape = shape  
        self.T = np.max(support) - np.min(support) # pulse length
        self.dt = abs(support[1] - support[0])
        if type(pulse_N) == type(None):
            self.pulse_N = self.support.size
        else:
            self.pulse_N = pulse_N

    def get_fdom(self):
        """
        Get corresponding frequency domain components
        """
#        df = round(1/self.T, 3)
#        fs = 1/self.dt 
        num_vals = self.support.size 
        freqs = np.fft.fftfreq(num_vals, d=self.dt)
#        freqs = np.arange(0, num_vals*df-df, df)
        fvals = np.fft.fft(self.shape)
        self.freqs = freqs
        self.fvals = fvals
        return freqs, fvals

    def plot_pulse(self):
        fig = plt.figure()
        plt.plot(self.support, self.shape)
        plt.xlabel("Time (s)")
        plt.ylabel("Pulse amplitude")
        plt.title("Pulse shape")
        return fig

    def plot_f_mag(self):
        fig = plt.figure()
        plt.scatter(self.freqs, abs(self.fvals))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Frequency domain")
        return fig

    def plot_f_phase(self):
        fig = plt.figure()
        plt.plot(self.freqs, np.angle(self.fvals)%(2*np.pi))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Angle (radians)")
        plt.title("Frequency domain phase")
        return fig

    def truncate_pulse(self, flow, fhigh,):
        """
        Cut down the pulse for computational needs
        flow - float
        fhigh - float
        Create new objects for the truncated frequencies, the truncated fft vals, and the indices of the original frequency set 
        Return them as well
        """
        freqs = self.freqs
        freq_trunc = np.array([x for x in freqs if (x > flow) and (x<fhigh)])
        inds = np.array([i for i in range(len(freqs)) if (freqs[i] > flow) and (freqs[i] < fhigh)])
        trunc_fft = self.fvals[inds] # corresponding to real freqs
        self.trunc_treqs = freq_trunc
        self.trunc_fvals = trunc_fft 
        self.trunc_inds = inds
        return freq_trunc, inds, trunc_fft

    def compare_truncation(self):
        """
        Compare time domain waveforms of truncated and untruncated lfm pulses
        """
        fvals = np.zeros(self.freqs.size)
        fvals[self.trunc_inds] = self.trunc_fvals
        td = np.fft.irfft(fvals)
        fig = plt.subplots(2,1)
        plt.suptitle('Compare truncated and original time domain pulses')
        plt.subplot(211)
        plt.plot(self.support, self.shape)
        plt.subplot(212)
        plt.plot(self.support, td)

    def stretch_signal(self, vs):
        """
        vs is the source speed on aline between source and receiver
        Dilate the signal waveform appropriately
        Positive vs is the source is moving toward the reciever (compressed)
        """
        alpha = vs/1500 # approximate stretching factor
        nsamps = self.support.size
        pulse_N =self.pulse_N
        N = nsamps # default size of new pulse
        """ If alpha > 1, there is a compression, so I need to make sure N is long enough to avoid wraparound  """
        if alpha > 1:
            if (1 + alpha)*(pulse_N-1)/pulse_N > N:
                N = int(np.power(2, np.log2(N)+1))
        """ Zero pad the signal and compute the FFT """
        sig = np.zeros(N, dtype=np.complex128)
        sig[:nsamps] = self.shape
        sf = np.fft.fft(sig)
        """ Compute W for inverse chirp Z """
        df = (1+alpha)/N
        """ Compute sequence a_k, b_k """
        square_vals = np.square(np.linspace(-N//2, N//2-1, N))
        ak = sf*np.exp(complex(0,1)*np.pi*df*square_vals)
        bk = np.exp(-complex(0,1)*np.pi*df*square_vals)
        """ Calculate convolution of ak and bk """
        Ak, Bk = np.fft.fft(ak), np.fft.fft(bk)
        prod = Ak*Bk
        tmp = np.fft.ifft(prod)/N
        chirp = tmp * bk.conj()
        chirp = hilbert(chirp.real) # make sure it's hilbert
        chirp = chirp.reshape(chirp.size) # cast to 1d array
        self.stretch_sig = chirp
        self.stretch_support = np.linspace(0, N*self.dt, N)
        return chirp
    
class PulseSynth:
    """ Perform a pulse synthesis using frequency domain or time domain models """
    def __init__(self, tdpulse, env):
        """
        Inputs
        tdpulse - TDPulse object
        model - Env object
        """
        self.tdpulse = tdpulse
        self.env = env

    def pop_env(self, zs, zr, dz, zmax, dr, rmax):
        """ 
        Give env field params
        """
        self.tdpulse.get_fdom() # populated freqs and fvals attributes
        self.env.add_source_params(self.tdpulse.freqs[0], zs, zr) # put in holder freq for now
        self.env.add_field_params(dz, zmax, dr, rmax)
        return

    def tt_synth(self, folder, fname, vs=0):
        """
        Perform a travel time synthesis with doppler
        If doppler is zero, do a delay and sum
        folder - string
            Example 'at_files/' (place to store generated at_files)
        fname - string
            Example 'deepwater_synth' template for saving the at files

        NOTE FIGURE OUT WHICH FREQUENCY TO RUN THE MODEL AT...
        """
        """
        Run Bellhop to get arrival structure
        """
        run_type='A'
        nbeams=231
        alpha=np.linspace(-20, 20, nbeams)
        box=Box(np.max(self.env.z_ss)+100, np.max(self.env.rmax*1e-3))
        deltas = 0
        beam = Beam(RunType=run_type, Nbeams=nbeams, alpha=alpha,box=box,deltas=deltas)
        self.env.freq = 10000
        self.env.run_model('bellhop', folder, fname, beam=beam)
        arrivals, pos = read_arrivals_asc(folder+fname)
        """
        Compute maximum channel spread to make sure my waveform doesn't
        get clipped
        """
        num_sources = len(arrivals)
        arrs = arrivals[1] # disregard the first one it corresponds to first range step
        position = index_to_pos(1, pos)
        print(position)
        num_arrivals = len(arrs)
        delays  = [x[1].real for x in arrs]
        spread = np.max(delays) - np.min(delays)
        """ Compute the stretched signals """
        stretched_sigs = []
        max_sig_N = 0
        for j in range(num_arrivals):
            arr = arrs[j]
            amp = arr[0]
            delay = arr[1].real
            src_angle = arr[2]
            print('src_angle', src_angle)
            proj_vel = vs*np.cos(np.pi*src_angle/180)
            sig = self.tdpulse.stretch_signal(proj_vel)
            sig_N = sig.size
            """
            Signals may be zero padded extra to ensure no wraparound
            in the chirp z-transform, so make sure we have space
            """
            if sig_N > max_sig_N:
                max_sig_N = sig_N
            stretched_sigs.append(sig)
        """
        Allocate array for spread signal
        """
        tmin = np.min(delays)
        print('First arrival time', tmin)
        print('spread', spread)
        min_N = int(spread/self.tdpulse.dt) + max_sig_N
        num_samples = min_N
        rcvd_signal = np.zeros(num_samples, dtype=np.complex128)
        t_grid = np.linspace(tmin, tmin+num_samples*self.tdpulse.dt, num_samples)
        for j in range(num_arrivals):
            arr = arrs[j]
            delay = arr[1].real
            ind = int((delay-tmin)/self.tdpulse.dt)
            curr_sig = stretched_sigs[j]
            rcvd_signal[ind:ind+curr_sig.size] += curr_sig
        return t_grid, rcvd_signal
        
    def synthesize(self, folder, model, fname, freqs, inds):
        """
        Run a model at each frequency  in freqs
        Weight each model by the appropriate fourier coefficient retrieved from the pulse transform
        Inverse transform the result
        Inputs -
        folder - string
        model - string
        freqs - np array
        inds - list 
        """
        rec_vals = np.zeros(self.tdpulse.freqs.size,dtype=np.complex128)
        i = 0
        for f in freqs:
            self.env.freq = f
            x, pos = self.env.run_model(model, folder, fname, zr_flag=True)
            rec_val = x[-1]
            weight = self.tdpulse.fvals[i]
            rec_vals[inds[i]] = rec_val*weight
            i += 1
        tx_pulse =  np.fft.ifft(rec_vals)
        return tx_pulse
        

        
