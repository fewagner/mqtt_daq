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


broker = 'localhost'
port = 10401
client_id = 'test-primary'
username = 'fwagner'
password = '1234'


# In[3]:


client_sender = connect_mqtt(broker, port, client_id, username, password, userdata={})


# In[4]:


test_msg = {'hello': 'world'}
# test_msg = 'hello world'


# In[5]:


while True:
    result = client_sender.publish('ccs/test', json.dumps(test_msg))
    check(result)
    time.sleep(5)


# In[ ]:




