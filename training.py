#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os
from utils import *


# In[2]:


nmbr_channels = 2
buffer_size = 10
model_name = 'dummy_agent'
steps = 100


# In[3]:


agents = []
for channel in range(nmbr_channels):
    agents.append(DummyAgent(channel, buffer_size))
    agents[-1].save('models/' + model_name + '_channel_{}.model'.format(channel))


# In[4]:


ph_mtime = [os.path.getmtime('data/ph_memory_channel_0.npy') for i in range(nmbr_channels)]
rms_mtime = [os.path.getmtime('data/ph_memory_channel_0.npy') for i in range(nmbr_channels)]


# In[5]:


# we can access the buffer with read rights
steps_counter = [0, 0]
while True:
    for channel in range(nmbr_channels):
        # train only if buffer changed
        current_ph_mtime = os.path.getmtime('data/ph_memory_channel_{}.npy'.format(channel))
        current_rms_mtime = os.path.getmtime('data/rms_memory_channel_{}.npy'.format(channel))
        if current_ph_mtime > ph_mtime[channel] or current_rms_mtime > rms_mtime[channel]:
            print('channel {}, training steps {}-{}'.format(channel, steps_counter[channel], steps_counter[channel] + steps))
            agents[channel].learn()
            agents[channel].save('models/' + model_name + '_channel_{}.model'.format(channel))
            ph_mtime[channel] = current_ph_mtime
            rms_mtime[channel] = current_rms_mtime
        steps_counter[channel] += steps
    time.sleep(2)


# In[ ]:




