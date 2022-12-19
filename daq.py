#!/usr/bin/env python
# coding: utf-8

# In[1]:


from paho.mqtt import client as mqtt_client
import numpy as np
import time
import matplotlib.pyplot as plt
import json
from utils import *


# In[2]:


broker = 'broker.hivemq.com'
port = 1883
client_id = 'daq-primary'
username = 'fwagner'
password = '1234'


# In[3]:


# daq control parameters

tpa = 1

nmbr_channels = 2
tpa_queue = list(range(10))
dac = np.random.uniform(-1, 1, size=nmbr_channels)
bias_current = np.random.uniform(-1, 1, size=nmbr_channels)


# In[4]:


client = connect_mqtt(broker, port, client_id, username, password)


# In[5]:


for i in range(nmbr_channels):
    subscribe(client, 'control/{}/set_control'.format(i))


# In[6]:


def receive_and_set(client, userdata, msg):
    
    try:
        channel = int(msg.topic.split('/')[1])
        
        data = json.loads(msg.payload)
        
        dac[channel] = data["DAC"]
        bias_current[channel] = data["BiasCurrent"]
    
    except KeyError as err_msg:
        print('KeyError: ', err_msg)
        pass


# In[7]:


client.on_message = receive_and_set


# In[8]:


# do some measurement

counter = 0

while True:
    for n in range(nmbr_channels):
        
        tpa = tpa_queue[counter]
        
        meas = np.random.normal(size=100, scale=.05)
        meas[25:50] += dac[n]*tpa*np.exp(-np.arange(25)/5)
        meas *= bias_current[n]
        
        payload = {
            "ChannelId": 3,
            "InjectedPulse": {
                "TPA": tpa,
                "nsTs": 4799090005531000,
                "nsTsUTC": 140196121154224
            },
            "LBaseline": 0.18045235450579847,  # TODO, optional
            "PulseHeight": 0.03339375199042272,  # TODO, optional
            "RMS": 0.03339375199042272,  # TODO, optional
            "DAC": dac[n], 
            "BiasCurrent": bias_current[n],
            "nsTsTrigger": 4799090007210000, # TODO
            "nsTsUTC": 1667840235744172000, # TODO
            "nsTsWindow": 4799090007210000, # TODO
            "BytesPerSample": 2, # TODO
            "Samples":  meas.tolist(),  # TODO as IntString
        }

        result = client.publish('trigger/{}/parameter'.format(n), json.dumps(payload))
        check(result)
        
        counter += 1
        counter %= len(tpa_queue)
        
        client.loop(1)
    time.sleep(2)


# In[ ]:




