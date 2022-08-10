import math
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from fitter import Fitter
import itertools
import scipy.stats
from fitter import Fitter
import pandas as pd

class CoordAng:
    def __init__(self, time, coord, ang):
        self.time=time
        self.coord = coord
        self.ang = ang


def readData(fileName):
    filePath = "D:\\Ipaper\\PythonCode"+"\\data_Oculus\\"
    f = open(filePath+ fileName)
    data = []
    Euler_data=[]
    time_data=[]
    for line in f:
        if line != 'Time x y z alpha beta gamma \n' and line != ' \n':
            line.strip('\n')
            coordinates = line.split(",")
            obj = CoordAng(float(coordinates[0]),[float(coordinates[1]), float(coordinates[2]), float(coordinates[3])], [float(coordinates[4]),float(coordinates[5]),float(coordinates[6])])
            data.append(obj)
            Euler_data.append(obj.ang)
            time_data.append(obj.time)
    return data, Euler_data, time_data


beingsaved = plt.figure()
#needs to be modified to fit yours
filePath = "D:\\Ipaper\\PythonCode"+"\\data_Oculus\\"
onlyfiles = [f for f in listdir(filePath) if isfile(join(filePath, f))]
index = 0
all_runs_list = []
head_Euler=[]
ori_time=[]
for f in onlyfiles:
    if '.meta' not in f and '.DS_Store' not in f:
        data,Euler_data,time_data = readData(f)
        index += 1
        all_runs_list.append(data)
        head_Euler.append(Euler_data)
        ori_time.append(time_data)

################Convert Euler to Polar and Azimuth Angles###########################
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
    temp2 = np.multiply(cos_fun(head_Euler[0]),cos_fun(head_Euler[1]))
    temp3 = np.multiply(np.multiply(cos_fun(head_Euler[1]),sin_fun(head_Euler[2])),sin_fun(head_Euler[0]))+\
    np.multiply(cos_fun(head_Euler[2]),sin_fun(head_Euler[1]))
    azimuth = np.arctan(temp3/temp2)
    azimuth = azimuth*180./np.pi
    if temp2<0:
        if temp3>0:
            azimuth = 180. + azimuth
        else:
            azimuth = -180. + azimuth
    return azimuth

#################Filters all measurements of a run that are at the beginning and in a vicinity of threshold_deg around the starting coordinates
def filter_starting(list_of_runs, time, threshold_deg=0.0):
    '''Filters all measurements of a run that are at the
    beginning and in a vicinity of threshold_deg around the starting coordinates.'''
    polar = []
    azimuth = []
    polar_run = []
    azimuth_run = []
    cleaned_polar = []
    cleaned_azimuth = []
    cleaned_polar_run = []
    cleaned_azimuth_run = []
    cleaned_time_run = []
    for num in range(len(list_of_runs)):
        polar_iter = []
        azimuth_iter = []
        run = list_of_runs[num]

        for run_iter in run:
            polar_iter.append(Euler_to_polar(run_iter))
            azimuth_iter.append(Euler_to_azimuth(run_iter))
        polar.extend(polar_iter)
        azimuth.extend(azimuth_iter)
        polar_run.append(polar_iter)
        azimuth_run.append(azimuth_iter)

        init_starting_polar = polar_iter[0]
        init_starting_azimuth = azimuth_iter[0]

        outside_bool = (np.absolute(init_starting_azimuth - azimuth_iter) > threshold_deg) & (
                    np.absolute(init_starting_azimuth - azimuth_iter) < 360. - threshold_deg) \
                       & (np.absolute(init_starting_polar - polar_iter) > threshold_deg)
        if np.any(outside_bool):
            first_left = np.amin(np.where(outside_bool))
            # cleaned_runs.append({key:value[first_left:] for key, value in run.items()})
            cleaned_polar.extend(polar_iter[first_left:])
            cleaned_polar_run.append(polar_iter[first_left:])

            cleaned_azimuth.extend(azimuth_iter[first_left:])
            cleaned_azimuth_run.append(azimuth_iter[first_left:])

            cleaned_time_run.append(time[num][first_left:])
    return cleaned_polar, cleaned_azimuth, polar_run, azimuth_run, polar, azimuth, cleaned_polar_run, cleaned_azimuth_run, cleaned_time_run

cleaned_polar, cleaned_azimuth, polar_run, azimuth_run, polar, azimuth, cleaned_polar_run, cleaned_azimuth_run, time = filter_starting(head_Euler, ori_time)

#############################(Optional)Filter out the repetitive data####################################################
'''
def filter_repetitive_data(list_of_angles, time):
    filtered_time=[]
    filtered_angles=[]
    for i in range(len(list_of_angles)):
        run = list_of_angles[i]
        filtered_run = [run[0]]
        filtered_time_run=[time[i][0]]
        for iter_ in range(1, len(run)):
            if run[iter_-1] == run[iter_]:
                r = np.random.uniform(low=0.0, high=1.0)
                if r>0.5:
                    filtered_time_run.pop()
                    filtered_time_run.append(time[i][iter_])
            else:
                filtered_run.append(run[iter_])
                filtered_time_run.append(time[i][iter_])
        filtered_time.append(filtered_time_run)
        filtered_angles.append(filtered_run)
    return filtered_angles, filtered_time
cleaned_polar_run, filtered_time = filter_repetitive_data(cleaned_polar_run, time)
cleaned_azimuth_run, filtered_time = filter_repetitive_data(cleaned_azimuth_run, time)
'''

############################Fit the Polar Angels########################################################
flat_polar = list(itertools.chain.from_iterable(cleaned_polar_run))
f = Fitter(flat_polar, xmin=0, xmax=180, bins=200, timeout=100, distributions = ["alpha","beta","chi", "chi2", "erlang", "expon", "exponpow", "genpareto", "genextreme", "gengamma", "gamma", "invgamma", "invgauss", "kappa3", "kappa4", "laplace", "levy", "logistic", "lognorm", "maxwell", "nakagami", "norm", "powerlaw", "rayleigh", "rice", "truncexpon", "uniform", "weibull_min", "weibull_max","wrapcauchy"])
#f = Fitter(flat_polar, xmin=0, xmax=180, bins=2000, timeout=100, distributions = ["laplace"])
f.fit()
f.get_best()
print(f.summary(Nbest=5, lw=2))

############################Plot the figure of polar angle############################################
flat_polar = list(itertools.chain.from_iterable(cleaned_polar_run))
param = scipy.stats.laplace.fit(flat_polar)
y = np.arange(0,180,1)
p = scipy.stats.laplace.pdf(y, *param[:-2], loc=param[-2], scale=param[-1])
n, bins, patches =plt.hist(flat_polar, range=(0, 180), density=True, color = 'skyblue', edgecolor = 'skyblue',bins = 180, label = 'Exp. data')
plt.plot(y, p, 'k', linewidth=4, label = 'Laplace fit')
plt.xlabel('Polar angle ($^ \circ$)',fontsize=23)
plt.ylabel('PDF',fontsize=23)
plt.legend(loc='upper right',bbox_to_anchor=(1.02, 1.03),fontsize=17,ncol=1)
plt.xticks([0,30,60,90,120,150,180],fontsize=23)
plt.yticks(fontsize=23)
plt.xlim([0,180])
plt.show()
beingsaved.savefig('polar_fit_our_dataset.eps',  bbox_inches = 'tight', format='eps', dpi=None)

#########################Fit the polar angle change########################################################
def uneven_to_even(list_of_angles, time,  time_slot = [0, 30, 1800]):
    start_time = time_slot[0]
    end_time = time_slot[1]
    bins = time_slot[2]
    slot = (end_time-start_time)/bins
    even_angles = []
    for j in range(12001):
        even_angles.append([])
    for i in range(len(list_of_angles)):
        print(i)
        run = list_of_angles[i]
        for iter_ in range(len(run)):
            for dif in range(len(run)-iter_):
                time_difference = int((time[i][iter_ + dif] - time[i][iter_])/slot+0.5)
                res = time[i][iter_ + dif] - time[i][iter_]-time_difference*slot;
                value = 600
                if (abs(res) < 0.5*slot) and ((time_difference==600)):
                    if run[iter_ + dif] - run[iter_] > 180:
                        even_angles[time_difference].append(run[iter_ + dif] - run[iter_] - 360)
                    elif run[iter_ + dif] - run[iter_] < -180:
                        even_angles[time_difference].append(run[iter_ + dif] - run[iter_] + 360)
                    else:
                        even_angles[time_difference].append(run[iter_ + dif] - run[iter_])
                if time_difference>value:
                    break
    return even_angles

even_azimuth = uneven_to_even(cleaned_polar_run, time,  time_slot = [0, 30, 1800])

all_polar = even_azimuth[600]
param = scipy.stats.laplace.fit(all_polar)
y = np.arange(-180,180,1)
p = scipy.stats.laplace.pdf(y, *param[:-2], loc=param[-2], scale=param[-1])


beingsaved = plt.figure()
f = Fitter(all_polar, xmin=-180, xmax=180, bins=180, timeout=100, distributions = ["alpha","beta","chi", "chi2", "erlang", "expon", "exponpow", "genpareto", "genextreme", "gengamma", "gamma", "invgamma", "invgauss", "kappa3", "kappa4", "laplace", "levy", "logistic", "lognorm", "maxwell", "nakagami", "norm", "powerlaw", "rayleigh", "rice", "truncexpon", "uniform", "weibull_min", "weibull_max","wrapcauchy"])
#f = Fitter(all_polar, xmin=-180, xmax=180, bins=100, timeout=100, distributions = ["laplace"])
f.fit()
f.summary(Nbest=10, lw=2)
n, bins, patches =plt.hist(all_polar, density=True, color = 'skyblue', edgecolor = 'skyblue',bins = 400, label = 'Exp. data')
#plt.plot(y, p, 'k', linewidth=4, label = 'Laplace')
plt.xlabel('$\Delta \phi$ ($^ \circ$)',fontsize=23)
plt.ylabel('PDF',fontsize=23)
plt.legend(loc='upper right',bbox_to_anchor=(1.02, 1.03),fontsize=17,ncol=1)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.show()
beingsaved.savefig('polar_change.eps',  bbox_inches = 'tight',format='eps',dpi=None)

#########################Fit the azimuth angle change########################################################
beingsaved = plt.figure()
even_azimuth = uneven_to_even(cleaned_azimuth_run, time,  time_slot = [0, 30, 1800])

all_polar = even_azimuth[600]
param = scipy.stats.laplace.fit(all_polar)
y = np.arange(-180,180,1)
p = scipy.stats.laplace.pdf(y, *param[:-2], loc=param[-2], scale=param[-1])

from fitter import Fitter
f = Fitter(all_polar, xmin=-180, xmax=180, bins=180, timeout=100, distributions = ["alpha","beta","chi", "chi2", "erlang", "expon", "exponpow", "genpareto", "genextreme", "gengamma", "gamma", "invgamma", "invgauss", "kappa3", "kappa4", "laplace", "levy", "logistic", "lognorm", "maxwell", "nakagami", "norm", "powerlaw", "rayleigh", "rice", "truncexpon", "uniform", "weibull_min", "weibull_max","wrapcauchy"])
#f = Fitter(all_polar, xmin=-180, xmax=180, bins=100, timeout=100, distributions = ["laplace"])
f.fit()
f.summary(Nbest=10, lw=2)
n, bins, patches =plt.hist(all_polar, density=True, color = 'skyblue', edgecolor = 'skyblue',bins = 400, label = 'Exp. data')
#plt.plot(y, p, 'k', linewidth=4, label = 'Laplace')
plt.xlabel('$\Delta \phi$ ($^ \circ$)',fontsize=23)
plt.ylabel('PDF',fontsize=23)
plt.legend(loc='upper right',bbox_to_anchor=(1.02, 1.03),fontsize=17,ncol=1)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.show()
beingsaved.savefig('azimuth_change.eps',  bbox_inches = 'tight',format='eps',dpi=None)


beingsaved = plt.figure()
all_polar = even_azimuth[600]
n, bins, patches =plt.hist(all_polar, density=True, color = 'skyblue', edgecolor = 'skyblue',bins = 200, label = 'Exp. data')

#def func(x, p, u, u1, b, sigma):
#    return p/(2*b)*np.exp(-1/b*np.absolute(x-u))/(1-np.exp((-180-u)/b))+(1-p)/sigma/np.sqrt(2*np.pi)*np.exp(-1/2*((x-u1)/sigma)**2)/0.5/(scipy.special.erf((180-u1)/np.sqrt(2)/sigma)-scipy.special.erf((-180-u1)/np.sqrt(2)/sigma))

def func(x, p, u, u1, b, b2):
    return p/(2*b)*np.exp(-1/b*np.absolute(x-u))/(1-np.exp((-180-u)/b))+(1-p)/(b2)*np.exp(-1/b2*np.absolute(x-u1))/(1+np.exp(-1/b2*np.absolute(x-u1)))**2/(2*(1+np.exp(-1/b2*np.absolute(180-u1)))**(-1)-1)

#def func(x,b):
#    return 1/(2*b)*np.exp(-1/b*np.absolute(x))/(1-np.exp(-180/b))

#def func(x, p, u,u1, sigma, sigma1):
#    return p/sigma1/np.sqrt(2*np.pi)*np.exp(-1/2*((x-u1)/sigma1)**2)/0.5/(scipy.special.erf((180-u1)/np.sqrt(2)/sigma1)-scipy.special.erf((-180-u1)/np.sqrt(2)/sigma1))+(1-p)/sigma/np.sqrt(2*np.pi)*np.exp(-1/2*((x-u)/sigma)**2)/0.5/(scipy.special.erf((180-u)/np.sqrt(2)/sigma)-scipy.special.erf((-180-u)/np.sqrt(2)/sigma))

#def func(x, p, u, b, u1, b2):
#    return p/(2*b)*np.exp(-1/b*np.absolute(x-u))/(1-np.exp(-180/b))+(1-p)/(2*b2)*np.exp(-1/b2*np.absolute(x-u))/(1-np.exp(-180/b2))

xdata=np.arange(-180,180,1.8)
popt, pcov = curve_fit(func, xdata, n, bounds=([0.0, -1,-1, 0. ,0.], [1, 1 ,1 ,np.inf, np.inf]))
#popt, pcov = curve_fit(func, xdata, n)
print(popt)
plt.plot(xdata, func(xdata, *popt),"k",linewidth=4, label = 'Logistic\n+Laplace')
plt.xlabel('$\Delta \phi$ ($^ \circ$)',fontsize=30)
plt.ylabel('PDF',fontsize=23)
plt.legend(loc='upper right',bbox_to_anchor=(1.02, 1.03),fontsize=18.5,ncol=1)
plt.xticks([-180,-90,0,90,180],fontsize=23)
plt.yticks(fontsize=23)
plt.xlim([-180,180])
plt.show()
beingsaved.savefig('azimuth_change.eps',  bbox_inches = 'tight',format='eps',dpi=None)
plt.show()
error=n-func(xdata, *popt)
print(sum([i*i for i in error])*1.8)
from scipy.stats import kurtosis
print(kurtosis(all_polar))

#######################################Interpolate the Data###########################################
from scipy.interpolate import interp1d
import itertools

def interpolar(filtered_time, cleaned_run, fillvalue=-10.):
    inter_polar_run = []
    inter_time_run = []
    for i in range(len(filtered_time)):
        f = interp1d(np.hstack(filtered_time[i]),cleaned_run[i],kind='cubic',bounds_error=False, fill_value=fillvalue)
        newx = np.linspace(0, 300, num=36000, endpoint=True)
        run = f(newx)
        eff_run = []
        eff_time =[]
        for j in range(len(run)):
            if run[j] != fillvalue:
                eff_run.append(run[j])
                eff_time.append(newx[j])
        inter_polar_run.append(eff_run)
        inter_time_run.append(eff_time)
    return inter_polar_run, inter_time_run
inter_polar_run, inter_time_run = interpolar(time, cleaned_polar_run)
inter_azimuth_run, inter_time_run = interpolar(time, cleaned_azimuth_run, fillvalue=-190.)


####################################Compute the ACF####################################################


def get_acf(inter_run):
    acf=np.zeros(10000)
    num=np.zeros(10000)
    i=-1
    for run in inter_run:
        i=i+1
        s = pd.Series(run)
        for j in range(10000):
            if len(run)>j:
                acf[j] = acf[j]+(s.autocorr(lag=j))*(len(inter_polar_run)-j)
                num[j] = num[j]+len(inter_polar_run)-j
    for j in range(10000):
        acf[j]=acf[j]/num[j]
    return acf
beingsaved = plt.figure()
acf1 = get_acf(inter_polar_run)
acf2 = get_acf(inter_azimuth_run)
#plt.figure(num=None, figsize=(4, 3), dpi=150, facecolor='w')
l1, = plt.plot(np.linspace(0,83.333,10000), acf1, color='k', linewidth=4, label='Polar angle')
l2, = plt.plot(np.linspace(0,83.333,10000), acf2, color='k', ls=':',alpha=0.8, linewidth=4, label = 'Azimuth angle')
plt.legend(loc='upper right',bbox_to_anchor=(1.03, 1.03),fontsize=25)
plt.xlabel('Time (s)',fontsize=25)
plt.ylabel('ACF',fontsize=25)
plt.axis([0, 60, -0.2, 1])
plt.grid(True)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
beingsaved.savefig('ACF_our_dataset.eps',  bbox_inches = 'tight',format='eps',dpi=None)