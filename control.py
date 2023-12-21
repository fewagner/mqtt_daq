#!/usr/bin/env python
# coding: utf-8

# In[1]:


from paho.mqtt import client as mqtt_client
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import os
import json
from utils import *


# In[2]:


broker = 'localhost'  # 'broker.hivemq.com'
port = 10401
client_id = 'control-secondary'
username = 'fwagner'
password = '1234'


# In[3]:


nmbr_channels = 2
buffer_size = 10
model_name = 'dummy_agent'
dac = np.random.uniform(-1, 1, size=nmbr_channels)
bias_current = np.random.uniform(-1, 1, size=nmbr_channels)


# In[4]:


# state_memories = [np.memmap('data/state_memory_channel_{}.npy'.format(i), dtype=float, shape=(1000000, 5)) for i in range(nmbr_channels)]
# next_state_memories = [np.memmap('data/next_state_memory_channel_{}.npy'.format(i), dtype=float, shape=(1000000, 5)) for i in range(nmbr_channels)]
# action_memories = [np.memmap('data/action_memory_channel_{}.npy'.format(i), dtype=float, shape=(1000000, 2)) for i in range(nmbr_channels)]
# reward_memories = [np.memmap('data/reward_memory_channel_{}.npy'.format(i), dtype=float, shape=(1000000,)) for i in range(nmbr_channels)]
# terminal_memories = [np.memmap('data/terminal_memory_channel_{}.npy'.format(i), dtype=float, shape=(1000000,)) for i in range(nmbr_channels)]

ph_memories, rms_memories, memory_idx = [], [], []

for i in range(nmbr_channels):
    mode = 'r+' if os.path.isfile('data/ph_memory_channel_{}.npy'.format(i)) else 'w+'
    ph_memories = [np.memmap('data/ph_memory_channel_{}.npy'.format(i), dtype=float, shape=(buffer_size,), mode=mode) for i in range(nmbr_channels)]
    
    mode = 'r+' if os.path.isfile('data/rms_memory_channel_{}.npy'.format(i)) else 'w+'
    rms_memories = [np.memmap('data/rms_memory_channel_{}.npy'.format(i), dtype=float, shape=(buffer_size,), mode=mode) for i in range(nmbr_channels)]
    
    mode = 'r+' if os.path.isfile('data/memory_idx_channel_{}.npy'.format(i)) else 'w+'
    memory_idx = [np.memmap('data/memory_idx_channel_{}.npy'.format(i), dtype=int, shape=1, mode=mode) for i in range(nmbr_channels)]


# In[5]:


client = connect_mqtt(broker, port, client_id, username, password, userdata={})


# In[6]:


for i in range(nmbr_channels):
    subscribe(client, 'trigger/{}/parameter'.format(i))


# In[7]:


def receive_and_respond(client, userdata, msg):
    
    try:
        # get data
        channel = int(msg.topic.split('/')[1])
        data = json.loads(msg.payload)

        dac[channel] = float(data["DAC"])
        bias_current[channel] = float(data["BiasCurrent"])

        samples = np.array(data["Samples"], dtype=float)  # TODO convert from Int16

        # calc features
        offset = np.mean(samples[:25])
        ph = np.max(samples[25:] - offset)
        rms = np.std(samples[:25])

        # write to buffer
        ph_memories[channel][memory_idx[channel][0]] = ph
        rms_memories[channel][memory_idx[channel][0]] = rms
        memory_idx[channel][0] += 1
        memory_idx[channel][0] %= buffer_size
        ph_memories[channel].flush()
        rms_memories[channel].flush()
        memory_idx[channel].flush()

        # plot 
        # plt.plot(samples)
        # plt.title('{}, {}'.format(msg.topic, datetime.now()))
        # plt.xlabel('Sample index')
        # plt.ylabel('Volt')
        # plt.show()

        # get new control data
        with open('models/' + model_name + '_channel_{}.model'.format(channel), 'rb') as handle:
            policy = pickle.load(handle)
        state = np.array([dac[channel], bias_current[channel], ph, rms])
        new_dac, new_bias_current = policy.predict(state)

        # respond
        print('memory_idx: {}, channel: {}, ph: {}, rms: {}, new dac: {}, new bias current: {}'.format(memory_idx[channel][0], channel, np.max(samples), np.std(samples), new_dac, new_bias_current))

        payload_response = {
            "ChannelId": channel,
            "nsTsUTC": time.time(),  # TODO is not in ns
            "DAC": new_dac, 
            "BiasCurrent": new_bias_current,
        }

        result = client.publish('control/{}/set_control'.format(channel), json.dumps(payload_response))
        check(result)
        
    except KeyError as err_msg:
        print('KeyError: ', err_msg)
        pass


# In[8]:


client.on_message = receive_and_respond


# In[9]:


client.loop_forever()


# In[ ]:




