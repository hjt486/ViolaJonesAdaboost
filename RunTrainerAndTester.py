#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2019

@author: Jiatai Han

This is to run trainer and test the trainer using test dataset

"""
from utils import *

# Set maximum window size here
window_size = 8
# Generate training Data
faces_images_count, non_faces_images_count = TrainData()
# Initial training process
# Set 2nd argument to 'new' to generate new features, otherwise load saved feature file from 'features.pk;'
TrainDetector(10, 'load', faces_images_count, non_faces_images_count, window_size)
# Generate test Data
faces_images_count, non_faces_images_count = TestData()
# Initial testing process
TestDetector(faces_images_count, non_faces_images_count)