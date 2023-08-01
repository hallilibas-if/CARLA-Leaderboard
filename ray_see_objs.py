from pynput import keyboard
import memcache
import cv2
import ray
import numpy as np

ray.init(address='auto')
sard_buffer = ray.get_actor(name="some_name", namespace="colors")
while True:
    print("Ray Liste :",ray.get(sard_buffer.get_actions.remote())[0])  
