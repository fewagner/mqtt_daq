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


broker = 'broker.hivemq.com'
port = 1883
client_id = 'test-secondary'
username = 'fwagner2'
password = '12345'


# In[3]:


client_receiver = connect_mqtt(broker, port, client_id, username, password)


# In[8]:


subscribe(client_receiver, 'ccs/#')


# In[9]:


def receive(client, userdata, msg):
    # print(time.time(), json.loads(msg.payload))
    print(time.time(), msg.payload)


# In[10]:


client_receiver.on_message = receive


# In[ ]:


client_receiver.loop_forever()


# In[ ]:




