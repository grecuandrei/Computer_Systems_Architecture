#!/bin/bash

module load libraries/cuda

make && 
python3 bench.py