from time import sleep
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
import carla

from PIL import Image, ImageDraw
import numpy as np
import cv2
#import detector.demo_yolopv2_post_onnx as detector #Install it from the official repo: https://github.com/conda-forge/onnxruntime-feedstock
#import copy
from pynput import keyboard
import ray   
ray.init(address='auto') 
#ray.init(address="137.226.131.47:56429") # when on same mashine
sard_buffer = ray.get_actor(name="carla_com", namespace="rllib_carla")       


def get_entry_point():
    return 'MyAgent'

class MyAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file="", *args, **kwargs):
        self.image = np.zeros((182,182,3))
        self.image_record = np.zeros((182,182,3))
        self.rewards = dict()
        self.scalarInput = []
        self.done = False
        self.ep_steps = 0
        
        #self.detector = detector.Mydetector()
        #input_size = "192,320"
        #self.input_size = [int(i) for i in input_size.split(',')]
      
        super().__init__(path_to_conf_file,*args, **kwargs)

    def setup(self, path_to_conf_file):
        #self.showImage("setup")
        print("I am in the setup function")

    def sensors(self):
            return [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.9, 'y': 0.0, 'z': 1.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': 182, 'height': 182, 'fov': 100,
                        'id': 'rgb'
                        },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': -2.5, 'y': 0.0, 'z': 2.3,
                        'roll': 0.0, 'pitch': -0.5, 'yaw': 0.0,
                        'width': 300, 'height': 300, 'fov': 90,
                        'id': 'rgb_record'
                        }
                    ]

    def run_step(self, input_data, reward, timestamp,idAgent, key):
        """
        Execute one step of navigation.
        :return: control
        """
        self.idAgent=idAgent
        self.key = key
        image = np.array(input_data['rgb'])
        image1= Image.fromarray(image[1])
        self.image  = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2RGB)

        """
        image_record = np.array(input_data['rgb_record'])
        image_record= Image.fromarray(image_record[1])
        self.image_record  = cv2.cvtColor(np.array(image_record), cv2.COLOR_BGR2RGB)
        self.showImage()
        """
        control = carla.VehicleControl()
        
        
        """
        image2 = copy.deepcopy(np.array(image1))
        draw_debug  = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2RGB)
        drivable_area, lane_line, bboxes, scores, class_ids = self.detector.run_inference(self.input_size,image2, 0.5)
        # Draw
        draw_debug = self.draw_debug(
            draw_debug,
            drivable_area,
            lane_line,
            bboxes,
            scores,
            class_ids,
        )
        self.image = cv2.resize(draw_debug, (256,144))
        """        

        #For Keyboard control    
        self.rewards = dict()
        self.scalarInput = []
        self.done = False
        for criterion in reward:
            #print("Status: {} | Actual Value: {} | expected_value_success: {} | Test name : {} ".format(criterion.test_status, str(criterion.actual_value) ,criterion.expected_value_success, criterion.name ))
            self.rewards[str(criterion.name)] = criterion.actual_value
            if criterion._terminate_on_failure ==True and criterion.test_status == "FAILURE":
                self.done = True

        self.scalarInput.append(reward[5].speedLimit) 
        self.scalarInput.append(reward[5].current_speed)
        
        #print("Print I Agent {}: ".format(idAgent))
        action=ray.get(sard_buffer.get_actions.remote(self.idAgent,self.key,self.image,self.scalarInput,self.rewards,self.done))
        
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
 
        return control
    

    def showImage(self, name="map"):
        #cv2.imshow('map', np.uint8(input_data))
        cv2.imshow(name,self.image_record)
        cv2.waitKey(1)
    def draw_debug(self,
        debug_image,
        drivable_area,
        lane_line,
        bboxes,
        scores,
        class_ids,):
        # Draw:Drivable Area Segmentation
        # Not in Drivable Area
        bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
        bg_image[:] = (255, 0, 0)

        mask = np.where(drivable_area[0] > 0.5, 0, 1)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.75, mask_image, 0.25, 1.0)

        # Drivable Area
        bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
        bg_image[:] = (0, 255, 0)

        mask = np.where(drivable_area[1] > 0.5, 0, 1)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

        # Draw:Lane Line
        bg_image = np.zeros(debug_image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 255)

        mask = np.where(lane_line > 0.5, 0, 1)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

        # Draw:Traffic Object Detection
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
        return debug_image

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_record = cv2.cvtColor(input_data['rgb_record'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        

        return {
                'rgb': rgb,
                'rgb_record': rgb_record
                }



    def destroy(self):
        """
        ActorSpeedAboveThresholdTest does not properly send a FAILURE information. 
        Therefore we have to check if the evaluation is over and if done is not set correctly (in the destroy function done have to be always True). 
        The weak point of this solution is that the Obs and Reward are sent twice.
        """
        print("I am in the distroy function")
        print("Key for {} is {} and will be also distroyed".format(self.idAgent, self.key))
        if self.done == False:
            self.done = True      
            #self.showImage("destroy")
            _=ray.get(sard_buffer.get_actions.remote(self.idAgent,self.key, self.image,self.scalarInput,self.rewards,self.done)) #The extra setted done flag should be send afterwards to rllib buffer
        else:
            pass