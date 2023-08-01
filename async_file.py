from time import sleep
from pynput import keyboard
import cv2
import ray
import numpy as np



print(ray.init(address='auto', namespace="rllib_carla"))
action=str("w")

@ray.remote(num_cpus=2)
class buffer_com(object):
    def __init__(self):
        self.agent_obs = []
        self.agent_actions=[1] 
        self.rewards = [1]

    def get_actions(self, obs, rewards):
        self.agent_obs.append(obs)
        self.rewards.append(rewards)
        while len(self.agent_actions)==0:
            #print("wainting inside get_actions")
            sleep(0.05)
        actions=self.agent_actions
        self.agent_actions = []
        return actions

    def get_sards(self, actions):
        self.agent_actions.append(actions)
        while len(self.agent_obs)==0:
            #print("wainting inside get_sards")
            sleep(0.05)
        obs=self.agent_obs
        rewards=self.rewards
        self.agent_obs = []
        self.rewards = []
        #print("obs get_sards :", obs)
        print("obs get_sards :", rewards)
        return obs,rewards 

def showImage(input_data):
    #cv2.imshow('map', np.uint8(input_data))
    cv2.imshow('map',cv2.cvtColor(np.array(input_data), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

# Create an actor process.
#sard_buffer=buffer_com.options(name="some_name1",lifetime="detached").remote()
sard_buffer=buffer_com.options(name="carla_com",max_concurrency=2).remote()

def press_callback(key):
    action=str(key)
    get_obs = ray.get(sard_buffer.get_sards.remote(action))
    showImage(get_obs[0][0])

def release_callback(key):
    #print("get 1: ", ray.get(sard_buffer.get_actions.remote()))  
    pass

l = keyboard.Listener(on_press=press_callback,on_release=release_callback)
l.start()
l.join()