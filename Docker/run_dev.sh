#!/bin/bash

PROJECT_NAME="dsol"
IMAGE_NAME="${PROJECT_NAME}:noetic-ros"
DATA_PATH="/media/${USER}/zhipeng_usb/datasets"
DATA_PATH2="/media/${USER}/zhipeng_8t/datasets"
# Pick up config image key if specified
if [[ ! -z "${CONFIG_DATA_PATH}" ]]; then
    DATA_PATH=$CONFIG_DATA_PATH
fi



docker build -t $IMAGE_NAME -f "${HOME}/vscode_projects/${PROJECT_NAME}/catkin_ws/src/${PROJECT_NAME}/Docker/Dockerfile" .


xhost +local:root

docker run \
    -e DISPLAY \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${HOME}/vscode_projects/${PROJECT_NAME}/catkin_ws:/root/catkin_ws \
    -v ${DATA_PATH}:/root/datasets \
    -v ${DATA_PATH2}:/root/datasets2 \
    --privileged \
    --cap-add sys_ptrace \
    --runtime=nvidia \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# docker run --rm -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

xhost -local:root