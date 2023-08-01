from time import sleep
from pynput import keyboard
import memcache
import cv2
import ray
import numpy as np



print(ray.init(address='auto', namespace="colors"))
action=str("w")

@ray.remote(num_cpus=2)
class buffer_com(object):
    def __init__(self):
        self.agent_obs = [1]
        self.agent_actions=[] 

    def get_actions(self, obs):
        self.agent_obs.append(obs)
        print("obs get_actions: ", self.agent_obs)
        if len(self.agent_actions)==0:
            return False
        actions=self.agent_actions
        self.agent_actions = []
        print("actions get_actions: ", actions)
        return actions

    def get_sards(self, actions):
        self.agent_actions.append(actions)
        print("actions get_sards:", self.agent_actions)
        if len(self.agent_obs)==0:
            return False
        obs=self.agent_obs
        self.agent_obs = []
        print("obs get_sards :", obs)
        return obs 

@ray.remote(num_cpus=1)
def imitate_leaderboard(sard_buffer):
    while True:
        get_actions=ray.get(sard_buffer.get_actions.remote(4))
        sleep(0.1)
        print("actions post main: ", get_actions) 

@ray.remote(num_cpus=1)
def imitate_keyboard(sard_buffer):
    while True:
        get_sards=ray.get(sard_buffer.get_sards.remote("'w'"))
        sleep(0.1)
        print("sards post main: ", get_sards) 

# Create an actor process.
#sard_buffer=buffer_com.options(name="some_name1",lifetime="detached").remote()
sard_buffer=buffer_com.options(name="some_name").remote()

def press_callback(key):
    action=str(key)
    get_obs = ray.get(sard_buffer.get_sards.remote(action))
    sleep(0.05)
    print("sards post main: ", get_obs)  
    #shared.set('Value', str(key))

def release_callback(key):
    #print("get 1: ", ray.get(sard_buffer.get_actions.remote()))  
    pass

def showImage(input_data):
    #cv2.imshow('map', np.uint8(input_data))
    cv2.imshow('map',cv2.cvtColor(np.array(input_data), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


imitate_leaderboard.remote(sard_buffer)
#imitate_keyboard.remote(sard_buffer)
#while True:
#    get_sards=ray.get(sard_buffer.get_sards.remote("'w'"))
#    sleep(0.1)
    #ray.wait(get_sards)
#    print("sards post main: ", get_sards) 
    #get_actions=ray.get(sard_buffer.get_actions.remote(4))
    #ray.wait(get_actions)
    #print("actions post main: ", get_actions) 

#imitate_keyboard.remote(sard_buffer)
"""
while True:
    get_sards=ray.get(sard_buffer.get_sards.remote("'w'"))
    ray.wait(get_sards)
    print("sards post main: ", get_sards) 
"""
#imitate_keyboard.remote(sard_buffer)
l = keyboard.Listener(on_press=press_callback,on_release=release_callback)
l.start()
l.join()

""" synchronous version 
while True:
    get_sards=ray.get(sard_buffer.get_sards.remote("'w'"))
    #ray.wait(get_sards)
    print("sards post main: ", get_sards) 
    get_actions=ray.get(sard_buffer.get_actions.remote(4))
    #ray.wait(get_actions)
    print("actions post main: ", get_actions) 
"""