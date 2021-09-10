import os
import csv
import numpy as np
import pygal

# change directory
os.chdir(r"C:\....\Predict_COVID")  # alter the directory !!!


DecDict = np.load("DecDict.npy", allow_pickle=True)
IncDict = np.load("IncDict.npy", allow_pickle=True)

decreaseDict = DecDict.item()
increaseDict = IncDict.item()

CodeLabel = np.load("CodeArray.npy", allow_pickle=True)
FileName = "COVID_map.svg"

wm = pygal.maps.world.World()
wm.title = 'COVID Increase or Decrease'
# Add an instance with population data as the base
# e.g. wm.add('Increase', {'ca': 0.5, 'us': 0.8, ....})
wm.add('Increase', increaseDict)
wm.add('Decrease', decreaseDict)
wm.render_to_file(FileName)

