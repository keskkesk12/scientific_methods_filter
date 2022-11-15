import numpy as np
from math import *
from utm import utmconv
import matplotlib.pyplot as plt


def dm2d_lat(dm):
  d = int(dm[0:2])
  m = float(dm[2:])
  return d + m/60.0

def dm2d_lon(dm):
  d = int(dm[0:3])
  m = float(dm[3:])
  return d + m/60.0


def main():

  coordinates = []
  dists = []
  encoder = []
  X = []

  data_file = open('static2.csv','r')
  data = data_file.readlines()

  acc = []
  

  uc = utmconv()
  # convert from geodetic to UTM
  x = np.array([0., 0.]) # [x, xd]
  A = np.array([[0., 1.],[0., 0.]])
  B = np.array([0., 1.])

  prev_t = 0
  bias = 0

  for i in range(len(data)):
    p = (data[i].strip()).split(",")
    label = p[0]

    # Predict
    if(label == "IMU"):
      t = float(p[1])
      a = float(p[2])
      enc = int(p[3])
      dt = float((t - prev_t))/1000.0
      x += dt*(A@x + B*(a-bias))
      encoder.append((t, -enc/120.0))
      X.append((t, x.copy()))
      prev_t = t
      acc.append((t, a))

    # Update
    else:
      t = int(p[1])
      latitude = dm2d_lat(p[2])
      longitude = dm2d_lon(p[3])
      (hemisphere, zone, letter, e, n) = uc.geodetic_to_utm (latitude, longitude)
      coordinates.append([e, n])
      dists.append((t, sqrt(e**2+n**2)-6165040.650308812))

  x_unzipped = list(zip(*X))
  acc_mean = np.mean(list(zip(*acc))[1])
  print(acc_mean)

  plt.plot(*list(zip(*acc)))
  plt.show()

if __name__ == "__main__":
  main()