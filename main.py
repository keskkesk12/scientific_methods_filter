import numpy as np
from math import *
from utm import utmconv
import matplotlib.pyplot as plt
from pathlib import Path
import csv

def getGPSSample(samples, t):
  i = 0
  while(samples[i][0] < t and i < len(samples)-1):
    i += 1
  i = max(i-1, 0)
  return samples[i]


def dm2d_lat(dm):
  d = int(dm[0:2])
  m = float(dm[2:])
  return d + m/60.0

def dm2d_lon(dm):
  d = int(dm[0:3])
  m = float(dm[3:])
  return d + m/60.0





def main():
  errors = []

  for file in Path(".").glob("1*.csv"):
    print(file)
    coordinates = []
    dists = []
    encoder = []
    X = []
    X_acc = []
    acc = []
    v_test = []

    data_file = open(file,'r')
    data = data_file.readlines()

    first_dist = 0

    uc = utmconv()
    # convert from geodetic to UTM
    x = np.array([0., 0.]) # [x, xd]
    x_acc = np.array([0., 0.]) # [x, xd]
    A_1 = np.array([[1., 0.0495],[0., 0.9802]])
    B_1 = np.array([0.001242, 0.0495])
    C_1 = np.array([1, 0])
    A_2 = np.array([[1., 0.8242],[0., 0.67]])
    B_2 = np.array([0.4395, 0.8242])
    C_2 = np.array([1, 0])
    P = np.array([[1, 0],[-1, 1]])
    Q = np.array([[1, 1.5],[0, 0.5]]) # process variance
    R = .2 # observation variance
    prev_t = 0
    bias = -0.27
    gps_cnt = 0

    for i in range(len(data)):
      p = (data[i].strip()).split(",")
      label = p[0]

      # Integrate ACC
      if(label == "IMU"):
        t = float(p[1])
        a = float(p[2])-bias
        x_acc = A_1@x_acc + B_1*a
        X_acc.append((t, x_acc.copy()))


      # Predict
      if(label == "IMU"):
        t = float(p[1])
        a = float(p[2])-bias
        enc = int(p[3])
        x = A_1@x + B_1*a
        encoder.append((t, -enc * 1.17/125.0))
        X.append((t, x.copy()))
        acc.append((t, a/10))
        v_test.append((t,x[1]))

      # Update
      else:
        t = int(p[1])
        latitude = dm2d_lat(p[2])
        longitude = dm2d_lon(p[3])
        (hemisphere, zone, letter, e, n) = uc.geodetic_to_utm (latitude, longitude)
        coordinates.append([e, n])
        d = sqrt(e**2+n**2)

        if(gps_cnt == 0):
          gps_cnt+=1
          first_dist = d
        d -= first_dist
        dists.append((t, d))

        # Update x
        yd = d - x[0]
        # L = A_2@P@C_2 * (1/((C_2@P@C_2 + R)))
        L = np.array([0.2,0.2])

        x += L * yd

        # Update P
        P = A_2@P@A_2.transpose() + Q - A_2@P@C_2 * (1.0/((C_2@P@C_2 + R))) * C_2@P@A_2.transpose()



    # Calculate errors
    error_acc = sum([abs(enc[1] - acc[1]) for (enc, acc) in zip(encoder, acc)])/len(encoder)
    error_gps = sum([abs(t[1] - getGPSSample(dists, t[0])[1]) for t in encoder])/len(encoder)
    error_filter = sum([abs(enc[1] - x[1][0]) for (enc, x) in zip(encoder, X)])/len(encoder)

    errors.append((error_acc, error_gps, error_filter))


    x_unzipped = list(zip(*X))
    x_acc_unzipped = list(zip(*X_acc))
    plt.plot(*list(zip(*dists)))
    # plt.plot(*list(zip(*v_test)))
    plt.plot(*list(zip(*encoder)))
    plt.plot(x_acc_unzipped[0], [x[0] for x in x_acc_unzipped[1]])
    plt.plot(x_unzipped[0], [x[0] for x in x_unzipped[1]])
    plt.legend(["GPS", "Encoder", "Accelerometer", "Prediction"])
    plt.draw()
    plt.waitforbuttonpress()
    plt.cla()
    
  with open('errors.csv', 'w', newline ='') as file:
    write = csv.writer(file)
    write.writerows(errors)

if __name__ == "__main__":
  main()