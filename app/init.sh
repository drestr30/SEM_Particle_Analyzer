#!/bin/bash
set -e

## docker creation
#docker build -t export2_img .

## run container
docker run --rm \
    --name export2_cont  \
    -p 8051:8051 \
    -v "$PWD":/code \
    export2_img

    #/bin/bash  #