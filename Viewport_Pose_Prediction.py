import math
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.misc import derivative
import scipy.integrate as integrate
from pynverse import inversefunc
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import Ridge
import time


def sin_fun(x):
    return np.sin(x* np.pi / 180.)
def cos_fun(x):
    return np.cos(x* np.pi / 180.)
def Euler_to_polar(head_Euler):
    '''Convert Euler to polar and azimuth representation.'''
    temp1 = np.multiply(sin_fun(head_Euler[2]),sin_fun(head_Euler[1]))-np.multiply(np.multiply(cos_fun(head_Euler[2]),cos_fun(head_Euler[1])),\
    sin_fun(head_Euler[0]))
    polar = np.arccos(temp1)
    polar = polar*180./np.pi
    return polar

def Euler_to_azimuth(head_Euler):
    '''Convert Euler to azimuth representation.'''
    temp2= np.multiply(cos_fun(head_Euler[0]),cos_fun(head_Euler[1]))
    temp3= np.multiply(np.multiply(cos_fun(head_Euler[1]),sin_fun(head_Euler[2])),sin_fun(head_Euler[0]))+\
    np.multiply(cos_fun(head_Euler[2]),sin_fun(head_Euler[1]))
    azimuth=np.arctan(temp3/temp2)
    azimuth = azimuth*180./np.pi
    if temp2<0:
        if temp3>0:
            azimuth = 180. + azimuth
        else:
            azimuth = -180. + azimuth
    return azimuth


class CoordAng:
    def __init__(self, time, coord, ang):
        self.time = time
        self.coord = coord
        self.ang = ang


def readData(fileName):
    filePath = os.getcwd() + "\\data_Desktop\\"
    # filePath needs to be modified according to your file directory
    f = open(filePath + fileName)

    Euler_data_per_traj = []
    time_data_per_traj = []
    position_data_per_traj = []
    polar_data_per_traj = []
    azimuth_data_per_traj = []
    for line in f:
        if line != 'Time x y z alpha beta gamma \n' and line != ' \n':
            line.strip('\n')
            coordinates = line.split(",")
            obj = CoordAng(float(coordinates[0]), [float(coordinates[1]), float(coordinates[2]), float(coordinates[3])],
                           [float(coordinates[4]), float(coordinates[5]), float(coordinates[6])])

            Euler_data_per_traj.append(obj.ang)
            time_data_per_traj.append(obj.time)
            position_data_per_traj.append(obj.coord)

            polar_data_per_traj.append(Euler_to_polar(obj.ang))
            azimuth_data_per_traj.append(Euler_to_azimuth(obj.ang))
    return Euler_data_per_traj, time_data_per_traj, position_data_per_traj, polar_data_per_traj, azimuth_data_per_traj




def PosePrediction(t, x, z, a, p):
    ######################################Parameters to tun##########################################
    history_window = 0.05
    prediction_window = 0.15
    # Convert list to numpy array for efficient operations if not already
    t = np.array(t)
    # Assuming new_x, new_z, new_a, new_p are initialized correctly to the right size and type
    new_x, new_z, new_a, new_p = np.copy(x), np.copy(z), np.copy(a), np.copy(p)

    t_start_idx = 0
    t_end_idx = np.argmax(t > t[t_start_idx] + history_window)
    t_forecast_idx = np.argmax(t > t[t_end_idx] + prediction_window)

    index = 0
    while t_forecast_idx < len(t) and t_end_idx < len(t):
        index += 1

        # Refit model for x
        model_x = Ridge().fit(t[t_start_idx:t_end_idx + 1].reshape(-1, 1), x[t_start_idx:t_end_idx + 1])
        new_x[t_end_idx + 1:t_forecast_idx + 1] = model_x.predict(
            t[t_end_idx + 1:t_forecast_idx + 1].reshape(-1, 1)).reshape(-1, 1)

        # Refit model for z
        model_z = Ridge().fit(t[t_start_idx:t_end_idx + 1].reshape(-1, 1), z[t_start_idx:t_end_idx + 1])
        new_z[t_end_idx + 1:t_forecast_idx + 1] = model_z.predict(
            t[t_end_idx + 1:t_forecast_idx + 1].reshape(-1, 1)).reshape(-1, 1)

        # Refit model for p
        model_p = Ridge().fit(t[t_start_idx:t_end_idx + 1].reshape(-1, 1), p[t_start_idx:t_end_idx + 1])
        new_p[t_end_idx + 1:t_forecast_idx + 1] = model_p.predict(
            t[t_end_idx + 1:t_forecast_idx + 1].reshape(-1, 1)).reshape(-1, 1)

        # Refit model for a
        model_a = Ridge().fit(t[t_start_idx:t_end_idx + 1].reshape(-1, 1), a[t_start_idx:t_end_idx + 1])
        new_a[t_end_idx + 1:t_forecast_idx + 1] = model_a.predict(
            t[t_end_idx + 1:t_forecast_idx + 1].reshape(-1, 1)).reshape(-1, 1)

        t_end_idx = t_forecast_idx
        t_start_idx = np.argmax(t > t[t_end_idx] - history_window)
        t_forecast_idx = np.argmax(t > t[t_end_idx] + prediction_window)
        if t_forecast_idx == 0 and not (t[0] > t[t_end_idx] + prediction_window):
            break  # Exit the loop if no further valid indices are found

    return new_x, new_z, new_a, new_p



filePath = os.getcwd() + "\\data_Desktop\\"  # needs to be modified to fit yours
onlyfiles = [f for f in listdir(filePath) if isfile(join(filePath, f))]
Euler_data = []
time_data = []
position_data = []
polar_data = []
azimuth_data = []
for f in onlyfiles:
    if '.meta' not in f and '.DS_Store' not in f:
        Euler_data_per_traj, time_data_per_traj, position_data_per_traj, polar_data_per_traj, azimuth_data_per_traj = readData(
                f)
        Euler_data.append(Euler_data_per_traj)
        time_data.append(time_data_per_traj)
        position_data.append(position_data_per_traj)
        polar_data.append(polar_data_per_traj)
        azimuth_data.append(azimuth_data_per_traj)

loss_m = []
loss_s = []
for idx in range(len(Euler_data)):

    t = time_data[idx]
    x = [sublist[0] for sublist in position_data[idx]]
    y = [sublist[1] for sublist in position_data[idx]]
    z = [sublist[2] for sublist in position_data[idx]]

    p = polar_data[idx]
    a = azimuth_data[idx]

    t = np.array(t).reshape((-1, 1))
    x = np.array(x).reshape((-1, 1))
    y = np.array(y).reshape((-1, 1))
    z = np.array(z).reshape((-1, 1))
    p = np.array(p).reshape((-1, 1))
    a = np.array(a).reshape((-1, 1))
    new_x, new_z, new_a, new_p = PosePrediction(t, x, z, a, p)

    for i in range(len(t)):
        loss_m.append(abs(x[i] - new_x[i]))
        loss_m.append(abs(z[i] - new_z[i]))
        loss_s.append(abs(p[i] - new_p[i]))
        loss_s.append(abs(a[i] - new_a[i]))



print(np.mean(loss_m))
print(np.mean(loss_s))




