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


flow = 2.5 * 1e3
fhigh = 4.5 * 1e3
pulse_width = .0150 # 150 ms pulse
pulse_length = 5
g_pulse = create_LFM_pulse(flow, fhigh, pulse_width, pulse_length)
plt.show()
g_pulse.plot_pulse()
g_pulse.get_fdom()
plt.figure()
plt.plot(g_pulse.support, np.fft.irfft(g_pulse.fvals))
g_pulse.plot_f_mag()
plt.show()

env_builder = factory.create('deepwater')
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
name = 'deepwater'




