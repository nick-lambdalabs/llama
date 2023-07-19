#!/usr/bin/env bash

pip install -e .
torchrun --nproc_per_node 1 chat.py