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

import sys
import time
from IPython.display import display, clear_output
import torch
from envs import OptimizationEnv
from sac import SoftActorCritic
from rlutils import ReturnTracker, ReplayBuffer

import pickle


# In[2]:


broker = 'localhost'
port = 10401
client_id = 'listener-secondary'
username = 'fwagner'
password = '1234'


# In[3]:


def receive_and_print(client, userdata, msg):
    
    try:
        print(time.time(), msg.topic)
        print(json.loads(msg.payload))
        print('\n')
        
        if msg.topic == 'ccs/trigger/samples':
            data = json.loads(msg.payload)
            print(type(data['Samples']))
            print(len(data['Samples']))
            
            with open('Samples.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except KeyError as err_msg:
        print('KeyError: ', err_msg)
        pass


# In[4]:


client = connect_mqtt(broker, port, client_id, username, password, userdata={})


# In[5]:


# subscribe(client, 'ccs/subscription/set')
# subscribe(client, 'ccs/subscription/ack')
# subscribe(client, 'ccs/trigger/samples')
# subscribe(client, 'ccs/control/set')
# subscribe(client, 'ccs/control/ack')

subscribe(client, 'trigger/#')
subscribe(client, 'control/#')


# In[6]:


client.on_message = receive_and_print


# In[7]:


client.loop_forever()


# In[ ]:




