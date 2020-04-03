import numpy as np
from matplotlib import pyplot as plt
from td_synth.lib import pulse_synth as ps
from scipy.signal import hilbert
import os
from env.env.envs import factory

'''
Description:
Test out signal stretching for doppler effect

Date: 
4/1/2020

Author: Hunter Akins
'''


"""
Make lfm signal
"""
flow = 2.5 *1e3
fhigh = 5.5 * 1e3
pulse_width = .12 # 100 ms

pulse = ps.create_LFM_pulse(flow, fhigh, pulse_width)

vs = 13.5 # source moving towards receiver is pos
pulse.stretch_signal(vs)

plt.figure()
plt.plot(pulse.support, (np.fft.fft(pulse.shape)))
plt.plot(pulse.stretch_support, (np.fft.fft(pulse.stretch_sig)))
plt.figure()
plt.plot(pulse.support, pulse.shape.real)
plt.plot(pulse.stretch_support, pulse.stretch_sig.real)

env_builder = factory.create('deepwater')
env=env_builder()
psynth = ps.PulseSynth(pulse, env)

zs,zr = 50, 50
rmax = 55*1e3
dr= 100
dz = 100
zmax = 5000

psynth.pop_env(zs, zr, dz, zmax, dr, rmax)
curr_dir = os.getcwd()
curr_dir += '/at_files/'
name = 'deepwater'


tgrid, rcvd_sig = psynth.tt_synth(curr_dir, name)
plt.figure()
plt.plot(tgrid, rcvd_sig)
plt.show()






