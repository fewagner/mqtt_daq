from paho.mqtt import client as mqtt_client
import numpy as np
import time
from tqdm.auto import tqdm, trange
import pickle

def connect_mqtt(broker, port, client_id, username, password, userdata):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id, userdata=userdata)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, topic):
    msg_count = 0
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1
        
def subscribe(client: mqtt_client, topic):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    client.subscribe(topic)
    client.on_message = on_message
    
class DummyAgent:
    
    def __init__(self, channel, buffer_size):
        self.channel = channel
        self.buffer_size = buffer_size
        self.ph_memory = np.memmap('data/ph_memory_channel_{}.npy'.format(channel), dtype=float, shape=(buffer_size,), mode='r')
        self.rms_memory = np.memmap('data/rms_memory_channel_{}.npy'.format(channel), dtype=float, shape=(buffer_size,), mode='r')
        self.memory_idx = np.memmap('data/memory_idx_channel_{}.npy'.format(channel), dtype=int, shape=1, mode='r')
    
    def learn(self, steps = 100):
        for step in trange(steps):
            
            batch_idx = np.random.randint(self.buffer_size, size=1)
            data = np.concatenate([self.ph_memory[batch_idx].reshape(-1,1), 
                                   self.rms_memory[batch_idx].reshape(-1,1)], axis=1)
            
            pass
    
    def predict(self, state):
        return np.random.uniform(-1, 1, size=2)
    
    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)
            
def check(result):
    if result[0] == 0:
        pass
    else:
        print('Message {} not sent!'.format(result[1]))