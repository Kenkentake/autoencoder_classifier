#!/bin/sh
. docker/env.sh
docker stop $CONTAINER_NAME
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p 7979:7979 \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=16g \
  $IMAGE_NAME
