#!/bin/bash

srun --time=02:00:00 --nodes=1 --cpus-per-task=8 --exclude=amp-1 --exclude=singularity --gres=gpu:1 --partition=gpu --pty bash
