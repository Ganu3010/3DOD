# from models.pnpp import *
# import torch
# from main import classify
# from data_loader import *
# import os
# # from flask import Flask, flash, request, redirect, url_for, send_from_directory
# from werkzeug.utils import secure_filename

# # img = np.load('areas/Area_1_conferenceRoom_1.npy'
# img = ['areas/']
# output = classify(img='areas/Area_1_conferenceRoom_1.npy', num_votes=1)

with open('predictions/output_1.txt') as f:
    for ech in f:
        print(ech, end='')
        