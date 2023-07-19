#!/usr/bin/env bash

# cd into the llama directory first
sudo docker run -it --rm -v $(pwd):/workspace/llama -w /workspace/llama bash run_chat_inside_docker.sh