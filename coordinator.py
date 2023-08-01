from time import sleep
import sys
import os
from subprocess import call
import subprocess
#https://stackoverflow.com/questions/13745648/running-bash-script-from-within-python
#https://stackoverflow.com/questions/35685403/python-run-subprocess-for-certain-time
# go into /var and check occupied space : sudo du -cha --max-depth=1 | grep -E "M|G"
# delete occupied space by docker: docker system prune -a

min=12000
NUM_AGENT =8
GPU_ID = 0

docker1="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=2500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=0 -prefernvidia"
docker2="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=3500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia" 
docker3="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=4500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia"
docker4="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=5500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia"

while 1:
    for x in range(0,NUM_AGENT):
        os.system("pkill -f leaderboard_evaluator.py")
    
    os.system("docker kill $(docker ps -q)")
    sleep(3)
    os.system('sh clear.sh')
    print("sleeping is over")
    sleep(5)

    subprocess.Popen(docker1, shell=True)
    sleep(3)
    subprocess.Popen(docker2, shell=True)
    sleep(3)
    subprocess.Popen(docker3, shell=True)
    sleep(3)
    subprocess.Popen(docker4, shell=True)
    sleep(4)
   

    p1=subprocess.Popen("./run_agent_1.sh") #ToDo  run chmod +x run_agent_2.sh also the coordinatpr.py has to be a executable
    sleep(2)
    p2=subprocess.Popen("./run_agent_2.sh")
    sleep(2)
    p3=subprocess.Popen("./run_agent_3.sh")
    sleep(2)
    p3=subprocess.Popen("./run_agent_4.sh")
    sleep(min*60)
    print("timeout is over")

