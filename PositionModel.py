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
import itertools
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

class CoordAng:
    def __init__(self, time, coord, ang):
        self.time=time
        self.coord = coord
        self.ang = ang



def readData(fileName):
    filePath = "D:\\Ipaper\\PythonCode"+"\\data_Desktop\\"
    #filePath needs to be modified according to your file directory
    f = open(filePath+ fileName)
    data = []
    Euler_data=[]
    time_data=[]
    position_data=[]
    for line in f:
        if line != 'Time x y z alpha beta gamma \n' and line != ' \n':
            line.strip('\n')
            coordinates = line.split(",")
            obj = CoordAng(float(coordinates[0]),[float(coordinates[1]), float(coordinates[2]), float(coordinates[3])], [float(coordinates[4]),float(coordinates[5]),float(coordinates[6])])
            data.append([float(coordinates[0]),float(coordinates[1]), float(coordinates[2]), float(coordinates[3]), float(coordinates[4]),float(coordinates[5]),float(coordinates[6])])
            Euler_data.append(obj.ang)
            time_data.append(obj.time)
            position_data.append(obj.coord)
    return data, Euler_data, time_data,position_data

filePath = "D:\\Ipaper\\PythonCode"+"\\data_Desktop\\"  # needs to be modified to fit yours
onlyfiles = [f for f in listdir(filePath) if isfile(join(filePath, f))]

all_runs_list = []
head_Euler = []
ori_time = []
position = []
f_list = []
for f in onlyfiles:
    if '.meta' not in f and '.DS_Store' not in f:
        f_list.append(f)
        data, Euler_data, time_data, position_data = readData(f)

        all_runs_list.append(data)
        head_Euler.append(Euler_data)
        ori_time.append(time_data)
        position.append(position_data)


######################################Extract Flights and Pause Time#################################################


dis = 0
d_thre = 0.1

def examine_condition1(run, idx):
    distance = math.sqrt(((run[idx][1] - run[idx + 1][1]) ** 2) + ((run[idx][3] - run[idx + 1][3]) ** 2))
    if distance > dis:
        result1 = 1
    else:
        result1 = 0
    return result1


def examine_condition2(run, x_flight_start_idx, x_flight_end_idx):
    result2 = 1
    p1 = np.array([run[x_flight_start_idx][1], run[x_flight_start_idx][3]])
    p2 = np.array([run[x_flight_end_idx][1], run[x_flight_end_idx][3]])
    for intermediate in range(x_flight_start_idx, x_flight_end_idx):
        p3 = np.array([run[intermediate][1], run[intermediate][3]])
        d = np.abs(np.cross(p2 - p1, p3 - p1)) / np.linalg.norm(p2 - p1)
        if d > d_thre:
            result2 = 0
            break
    return result2


def extract_flight_and_pausetime(run):
    pausetime = []
    flight = []
    x_flight_start_idx = 0
    x_flight_end_idx = 0
    flag_condition1 = 1
    for idx in range(len(run) - 1):
        if examine_condition1(run, idx) == 0:
            if flag_condition1 == 1:
                x_pause_start = run[idx][0]
                x_pause_end = run[idx + 1][0]

                if x_flight_start_idx != 0:
                    flight.append(run[x_flight_start_idx] + run[x_flight_end_idx])
            else:
                x_pause_end = run[idx + 1][0]
            flag_condition1 = 0

        else:
            if flag_condition1 == 0:
                pausetime.append([x_pause_start, x_pause_end])
                x_flight_start_idx = idx;
                x_flight_end_idx = idx + 1;
            else:

                if examine_condition2(run, x_flight_start_idx, x_flight_end_idx) == 1:
                    x_flight_end_idx = idx + 1;
                else:
                    flight.append(run[x_flight_start_idx] + run[x_flight_end_idx])
                    x_flight_start_idx = idx
                    x_flight_end_idx = idx + 1
            flag_condition1 = 1

    return flight, pausetime

flight=[]
pausetime=[]
for idx in range(len(all_runs_list)):
        print(idx)
        new_flight,new_pausetime=extract_flight_and_pausetime(all_runs_list[idx])
        flight.append(new_flight)
        pausetime.append(new_pausetime)

###############################Rectangular Model in Paper On the Levy-Walk Nature of Human Mobility###############################################
'''
beingsaved = plt.figure()

flight_length = []
pausetime_length = []
flight_time = []
flight_angle = []
for idx in range(len(flight)):
    run = flight[idx]
    flight_length_temp = []
    flight_time_temp = []
    flight_angle_temp = []
    for jdx in range(len(run)):
        length = math.sqrt(((run[jdx][1] - run[jdx][8]) ** 2) + ((run[jdx][3] - run[jdx][10]) ** 2))
        flight_time_temp.append((run[jdx][7] - run[jdx][0]))
        flight_length_temp.append(length)
        temp3 = run[jdx][1] - run[jdx][8]
        temp2 = run[jdx][3] - run[jdx][10]
        angle_temp = np.arctan(temp3 / temp2)
        angle_temp = angle_temp * 180. / np.pi
        if temp2 < 0:
            if temp3 > 0:
                angle_temp = 180. + angle_temp
            else:
                angle_temp = -180. + angle_temp
        flight_angle_temp.append(angle_temp)
    flight_length.append(flight_length_temp)
    flight_time.append(flight_time_temp)
    flight_angle.append(flight_angle_temp)

for idx in range(len(pausetime)):
    run = pausetime[idx]
    pausetime_length_temp = []
    for jdx in range(len(run)):
        length = run[jdx][1] - run[jdx][0]
        pausetime_length_temp.append(length)
    pausetime_length.append(pausetime_length_temp)

'''

###############################Angle Model in Paper On the Levy-Walk Nature of Human Mobility###############################
beingsaved = plt.figure()

flight_length = []
pausetime_length = []
flight_time = []
flight_angle = []
for idx in range(len(flight)):
    run = flight[idx]
    flight_length_temp = []
    flight_time_temp = []
    flight_angle_temp = []

    jdx = 0
    start_idx = jdx
    temp3 = run[jdx][1] - run[jdx][8]
    temp2 = run[jdx][3] - run[jdx][10]
    angle_temp = np.arctan(temp3 / temp2)
    angle_temp = angle_temp * 180. / np.pi
    if temp2 < 0:
        if temp3 > 0:
            angle_temp = 180. + angle_temp
        else:
            angle_temp = -180. + angle_temp

    previous_angle = angle_temp

    while jdx < len(run) - 1:
        jdx = jdx + 1
        temp3 = run[jdx][1] - run[jdx][8]
        temp2 = run[jdx][3] - run[jdx][10]
        angle_temp = np.arctan(temp3 / temp2)
        angle_temp = angle_temp * 180. / np.pi
        if temp2 < 0:
            if temp3 > 0:
                angle_temp = 180. + angle_temp
            else:
                angle_temp = -180. + angle_temp
        ###################angle threshold to be modified###############################
        if abs(angle_temp - previous_angle) > 5:

            end_idx = jdx - 1
            length = math.sqrt(
                ((run[start_idx][1] - run[end_idx][8]) ** 2) + ((run[start_idx][3] - run[end_idx][10]) ** 2))
            flight_time_temp.append((run[end_idx][7] - run[start_idx][0]))
            flight_length_temp.append(length)
            angle_model_angle_temp = np.arctan(
                (run[start_idx][1] - run[end_idx][8]) / (run[start_idx][3] - run[end_idx][10]))
            angle_model_angle_temp = angle_model_angle_temp * 180. / np.pi
            if run[start_idx][3] - run[end_idx][10] < 0:
                if run[start_idx][1] - run[end_idx][8] > 0:
                    angle_model_angle_temp = 180. + angle_model_angle_temp
                else:
                    angle_model_angle_temp = -180. + angle_model_angle_temp
            flight_angle_temp.append(angle_model_angle_temp)

            start_idx = jdx
            previous_angle = angle_temp

    end_idx = len(run) - 1

    length = math.sqrt(((run[start_idx][1] - run[end_idx][8]) ** 2) + ((run[start_idx][3] - run[end_idx][10]) ** 2))
    flight_time_temp.append((run[end_idx][7] - run[start_idx][0]))
    flight_length_temp.append(length)
    angle_model_angle_temp = np.arctan((run[start_idx][1] - run[end_idx][8]) / (run[start_idx][3] - run[end_idx][10]))
    angle_model_angle_temp = angle_model_angle_temp * 180. / np.pi
    if run[start_idx][3] - run[end_idx][10] < 0:
        if run[start_idx][1] - run[end_idx][8] > 0:
            angle_model_angle_temp = 180. + angle_model_angle_temp
        else:
            angle_model_angle_temp = -180. + angle_model_angle_temp
    flight_angle_temp.append(angle_model_angle_temp)

    flight_length.append(flight_length_temp)
    flight_time.append(flight_time_temp)
    flight_angle.append(flight_angle_temp)

for idx in range(len(pausetime)):
    run = pausetime[idx]
    pausetime_length_temp = []
    for jdx in range(len(run) - 1):
        length = run[jdx][1] - run[jdx][0]
        pausetime_length_temp.append(length)
    pausetime_length.append(pausetime_length_temp)

####################################################Pause Time Distribution###########################################

print(len(list(itertools.chain.from_iterable(pausetime_length))))

beingsaved = plt.figure()
flat = list(itertools.chain.from_iterable(pausetime_length))
for i in range(4786 - 1694):
    flat.append(0)

n, bins, patches = plt.hist(flat, density=True, color='skyblue', edgecolor='skyblue', bins=300, label='Exp. data')

lbd = 1 / 0.62610 * 0.3539
c = 0.3539
y = np.arange(0, 15, 0.1)
p = []
for Y in y:
    p.append((1 - c) * (Y == 0 or Y == 0.1) + c * lbd * math.exp(-lbd * Y))

plt.plot(y, p, 'k', linewidth=4, label='Exponential dist. \nand a "bump"')
plt.xlabel('Pause time (s)', fontsize=23)
plt.ylabel('PDF', fontsize=23)
plt.legend(loc='upper right', bbox_to_anchor=(1.02, 1.03), fontsize=19, ncol=1)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.ylim([0, 1])
plt.xlim([0, 10])
plt.show()
beingsaved.savefig('pausetime_distribution.eps', bbox_inches='tight', format='eps', dpi=None)


####################################################Flight Sample###########################################
beingsaved = plt.figure()
x=[]
y=[]
for idx in range(2,3):
    run=all_runs_list[idx]
    for jdx in range(len(run)):
        pose_sample=run[jdx]
        x.append(pose_sample[1])
        y.append(pose_sample[3])
plt.plot(x,y,linewidth=3,alpha=1,color="skyblue",label = 'Collected trajectory')

x=[]
y=[]

for idx in range(2, 3):
    run=flight[idx]
    for jdx in range(len(run)):
        pose_sample=run[jdx]
        x.append(pose_sample[1])
        y.append(pose_sample[3])
        x.append(pose_sample[8])
        y.append(pose_sample[10])
plt.plot(x,y,color='k',marker="o",linestyle='--',linewidth=3,markersize=4,label = 'Extracted flights')

plt.xlabel('$x$ (m)',fontsize=23)
plt.ylabel('$z$ (m)',fontsize=23)
plt.legend(loc='upper right',bbox_to_anchor=(1.02, 1.03),fontsize=18,ncol=1)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.ylim([-30,70])
plt.show()
beingsaved.savefig('flight_sample.eps',  bbox_inches = 'tight',format='eps',dpi=None)

#################################################Flight Time Distribution#########################################
flat = list(itertools.chain.from_iterable(flight_time))


def getCumulProb(flightLengths):
    '''
    *input:
        flightLengths: a list of flight lengths
    *functionality:
        plots cumulative probability distribution of the input data
    *output:
        a list of cumulative probabilities, each corresponding to a flight length in flightLengths
    '''
    numFlights = len(flightLengths)
    cumulProbs = []
    for length in flightLengths:
        cumulProbs.append(len([i for i in flightLengths if i <= length]) / numFlights)
    return cumulProbs


def optimizeLamb(flightLengths):
    '''
    *input: a list of flight lengths
    *output: the lambda value to be used in the function cumulDistFunc
    '''
    cumulProbs = getCumulProb(flightLengths)
    flightLengths = sorted(flightLengths)
    popt, pcov = curve_fit(cumulDistFunc, flightLengths, cumulProbs)
    return popt[0]


def cumulDistFunc(lengths, lamb):
    '''
    *inputs:
        lengths: a list of flight lengths
        lamb: the lambda value to be used in fitting the original cumulative probabilty distribution of flight lengths
            to the distribution derived from a cumulative probability distribution function
    *output: a list with outputs from the cumulative probability distribution function
    '''
    fittedLengths = []
    for length in lengths:
        fittedLengths.append(1 - math.exp((-1) * lamb * length))
    return fittedLengths


def Plot_Prob(flightLengths):
    '''
    *input:
        flightLengths: a list of flight lengths
    *functionality:
        plots cumulative probability distribution of the input data
    *output:
        a list of cumulative probabilities, each corresponding to a flight length in flightLengths
    '''
    numFlights = len(flightLengths)
    cumulProbs = []
    for length in flightLengths:
        cumulProbs.append(len([i for i in flightLengths if i <= length]) / numFlights)
    plt.plot(sorted(flightLengths), sorted(cumulProbs), linewidth=4, color='b', linestyle='-', label='Flight samples')


x = np.random.uniform(-20, 20, 5000)
y = np.random.uniform(-20, 20, 5000)
t = []
for i in range(4999):
    t.append(math.sqrt((x[i] - x[i + 1]) * (x[i] - x[i + 1]) + (y[i] - y[i + 1]) * (y[i] - y[i + 1])))


def getCumulProb2(flightLengths):
    '''
    *input:
        flightLengths: a list of flight lengths
    *functionality:
        plots cumulative probability distribution of the input data
    *output:
        a list of cumulative probabilities, each corresponding to a flight length in flightLengths
    '''
    flightLengths = [number / 1.5 for number in flightLengths]
    numFlights = len(flightLengths)
    cumulProbs = []
    # SortedFlightLengths = sorted(flightLengths)
    for length in flightLengths:
        cumulProbs.append(len([i for i in flightLengths if i <= length]) / numFlights)
    plt.plot(sorted(flightLengths), sorted(cumulProbs), linewidth=4, color='k', linestyle='--', label='Classical RWP')
    # plt.title('Flight Length CDF')
    # plt.show()
    return cumulProbs


beingsaved = plt.figure()
Plot_Prob(flat)
getCumulProb(flat)

fitted = sorted(cumulDistFunc(flat, 1 / 2.4924102486209914))
plt.plot(sorted(flat), fitted, linewidth=4, color='r', linestyle='-.', label='Paused-MRWP')

getCumulProb2(t)
plt.xlabel('Flight time (s)', fontsize=22)
plt.ylabel('CDF', fontsize=22)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5), fontsize=19, ncol=1)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim([0, 40])
plt.show()
beingsaved.savefig('VKflight.eps', bbox_inches='tight', format='eps', dpi=None)

##########################################The difference between the walking direction and the azimuth angle################
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


def calculate_walking_direction_and_azimuth(head_Euler, position):
    polar_run = []
    azimuth_run = []
    walking_direction = []
    difference = []

    for num in range(len(head_Euler)):
        polar_iter = []
        azimuth_iter = []
        head_run = head_Euler[num]

        walking_direction_iter = []
        pos_run = position[num]

        difference_iter = []
        for run_iter in range(1, len(head_run)):

            if pos_run[run_iter][2] != pos_run[run_iter - 1][2]:
                walking_temp = math.atan((pos_run[run_iter][0] - pos_run[run_iter - 1][0]) / (
                            pos_run[run_iter][2] - pos_run[run_iter - 1][2])) * 180. / np.pi
                if pos_run[run_iter][2] - pos_run[run_iter - 1][2] < 0:
                    if pos_run[run_iter][0] - pos_run[run_iter - 1][0] > 0:
                        walking_temp = 180. + walking_temp
                    else:
                        walking_temp = -180. + walking_temp
                walking_direction_iter.append(walking_temp)
                polar_iter.append(Euler_to_polar(head_run[run_iter]))
                azimuth_iter.append(Euler_to_azimuth(head_run[run_iter]))
                difference_iter.append(Euler_to_azimuth(head_run[run_iter]) - walking_temp)

        polar_run.append(polar_iter)
        azimuth_run.append(azimuth_iter)
        walking_direction.append(walking_direction_iter)
        difference.append(difference_iter)

    return polar_run, azimuth_run, walking_direction, difference

beingsaved = plt.figure()
polar_run, azimuth_run, walking_direction,difference = calculate_walking_direction_and_azimuth(head_Euler, position)
flat = list(itertools.chain.from_iterable(difference))
n, bins, patches =plt.hist(flat, density=True, color = 'skyblue', edgecolor = 'skyblue',bins = 1000, label = 'Exp. data')
plt.show()
beingsaved.savefig('correlation.eps', bbox_inches='tight', format='eps', dpi=None)