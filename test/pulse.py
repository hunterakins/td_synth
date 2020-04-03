import timing
import numpy as np
from matplotlib import pyplot as plt
from td_synth.lib.pulse_synth import TDPulse, PulseSynth, create_LFM_pulse
from env.env.envs import factory
import os

'''
Description:
Tests for pulse synthesis code


Author: Hunter Akins
'''


flow = 200
fhigh = 300
pulse_width = .2
pulse_length = 5
g_pulse = create_LFM_pulse(flow, fhigh, pulse_width, pulse_length)
g_pulse.plot_pulse()
g_pulse.get_fdom()
plt.figure()
plt.plot(g_pulse.support, np.fft.irfft(g_pulse.fvals))
g_pulse.plot_f_mag()

env_builder = factory.create('iso')
env=env_builder()
ps = PulseSynth(g_pulse, env)

zs,zr = 70, 50
rmax = 20*1e3
dr= 100
dz = .1
zmax = 100

ps.pop_env(zs, zr, dz, zmax, dr, rmax)
curr_dir = os.getcwd()
curr_dir += '/at_files/'
name = 'iso'

freq_trunc, inds, trunc_fft = g_pulse.truncate_pulse(100, 450)
g_pulse.compare_truncation()
ps.tt_synth(curr_dir, name)
synthed = ps.synthesize(curr_dir, 'kraken', name, freq_trunc, inds)
plt.figure()



