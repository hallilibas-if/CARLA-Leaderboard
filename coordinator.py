from time import sleep
import sys
import os
from subprocess import call
import subprocess
#https://stackoverflow.com/questions/13745648/running-bash-script-from-within-python
#https://stackoverflow.com/questions/35685403/python-run-subprocess-for-certain-time
# go into /var and check occupied space : sudo du -cha --max-depth=1 | grep -E "M|G"
# delete occupied space by docker: docker system prune -a

min=30
NUM_AGENT =4
GPU_ID = 1

docker1="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=2500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia"
docker2="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=3500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia" 
docker3="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=4500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia"
docker4="docker run --privileged --gpus device="+str(GPU_ID)+" --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.10.1 /bin/bash ./CarlaUE4.sh -quality-level=Epic -world-port=5500 -resx=400 -resy=300 -benchmark -fps 10 -graphicsadapter=1 -prefernvidia"

docker_list = [docker1,docker2,docker3,docker4]

while 1:
    for x in range(0,NUM_AGENT):
        os.system("pkill -f leaderboard_evaluator.py")
        sleep(2)
    
    os.system("docker kill $(docker ps -q)")
    sleep(3)
    os.system('sh clear.sh')
    print("sleeping is over")
    sleep(5)

    for x in range(0,NUM_AGENT):
        subprocess.Popen(docker_list[x], shell=True)
        sleep(3)
    
    sleep(2)

    for y in range(0,NUM_AGENT):
        _=subprocess.Popen("./run_agent_"+str(y+1)+".sh") #ToDo  run chmod +x run_agent_2.sh also the coordinatpr.py has to be a executable
        sleep(2)

    sleep(min*60)
    print("timeout is over")

