
#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
from collections import Counter
import math


def water_vel_incomp(p):
    rho = 1000;
    p_0 = 101325;
    v_w = np.sqrt(2*(p*6.89*10**6-p_0)/rho)
    return v_w;

def water_vel_comp(p_1):
    rho = 1000
    k_0 = 2.15*10**9;
    n = 7.15;
    p_0 = 101325;
    p_2 = p_0;
    p_1 = p_1*6.89*10**6;
    v_w = np.sqrt(2*k_0*(((1+(n/k_0)*(p_1-p_0))**(1-1/n))/(rho*n*(1-1/n))-((1+(n/k_0)*(p_2-p_0))**(1-1/n))/(rho*n*(1-1/n))))
    return v_w;

def tate_equation(p):
    rho = 1000;
    k_0 = 2.15*10**9;
    n = 7.15;
    p_0 = 101325;
    p = p*6.89*10**6
    
    rho_comp = rho*(1+(n/k_0)*(p-p_0))**(-1/n)
    return rho_comp;
    
def water_flow_theo(o,p,eqn_type):
    o = o*.0254
    a = np.pi*(o/2)**2
    rho = 1000;
    
    rho_comp = tate_equation(p);
    
    if eqn_type == 0: # compressible velocity equation
        wfr = rho_comp*a*water_vel_comp(p) #kg/s
        wfr = 2.204*60*wfr; #lb/min
    elif eqn_type == 1: # incompressible velocity equation
        wfr = rho*a*water_vel_comp(p) #kg/s
        wfr = 2.204*60*wfr; #lb/min    
    return wfr;

def water_flow_meas(o,p):
    wfr = (21101*o**2+436.59*o-2.8774)*(.1066*p**.5503);
    return wfr;

def hyd_pow(o,p):
    fr = water_flow_meas(o, p)
    p_hy = fr*(p*6.90*10**6)*(.0037/60/8.35)/745
    
    return p_hy

def r_0(mt):
    return 24.58*mt - .0735; 

def machine_num(mat):
    mn = {
        'Aluminum 2024' : 214,
        'Aluminum 6061' : 219,
        'Brass 260' : 155,
        'Brass 360' : 160,
        'Copper C110' : 103,
        'Steel A36' : 81.3,
        'Stainless Steel 304' : 80.8,
        'Stainless Steel 316' : 82.5,
        'Stone': 390,
    }
    return mn[str(mat)]

def kin_equ(s_0m, s_0b, r_sm, r_sb, psi_max, r_0, p_h, r, d_o):
    s_0 = s_0m*d_o + s_0b;
    r_s = r_sm*d_o + r_sb;

    return p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2

def opt_all(para, vals):
    s_0m, s_0b, r_sm, r_sb = para;
    psi_max, r_0, p_h, r, v_sep, d_o = vals

    s_0 = s_0m*d_o + s_0b;
    r_s = r_sm*d_o + r_sb;

    error = np.sum(((p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

    return error

def optimize_all(vals):
    psi_max, r_0, p_h, r, v_sep, d_o = vals;
    guess = np.array([1,1,1,1])
    inp = ([psi_max, r_0, p_h, r, v_sep, d_o])
    optim_all = sp.optimize.fmin(func = opt_all, x0 = guess, args = (inp,))
    return optim_all

file_name = 'separation_data.xlsx'
my_path = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + file_name;
data_sep = pd.read_excel(io = my_path, sheet_name = 'Sheet1')
data_sep.columns = ['Date',
                    'Sorting',
                    'Sort',
                    'Material',
                    'Thickness',
                    'Table',
                    'Machine',
                    'Pump',
                    'Nozzle',
                    'Abrasive',
                    'On/Off Valve',
                    'On/Off Valve Type',
                    'Cutting Head Number',
                    'Orifice',
                    'Mixing Tube Diameter',
                    'Mixing Tube Length',
                    'Flow Conditioner',
                    'Pressure',
                    'Water Flow Rate (gpm)',
                    'Desired Abrasive Feed Rate',
                    'Actual Abrasive',
                    'Material Separation Speed',
                    'Percentage 1',
                    'Percentage 2',
                    'Percentage 3',
                    'Experimental Separation',
                    'Abrasive Loading']             

#sort data into by material type, material thickness, mixing tube diameter, orifice diameter, pressure, and abrasive feed rate.
data_sep = data_sep.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])

data_filt = data_sep[data_sep['Material'] == 'Aluminum 6061']
data_filt = data_filt[data_filt['Experimental Separation'] < 500]
data_filt = data_filt[data_filt['Actual Abrasive'] < 5]
#data_filt = data_filt[data_filt['Pressure'] < 57]
data_filt = data_filt[data_filt['Thickness'] < 1.25]
data_filt = data_filt[data_filt['Thickness'] > .9]
data_filt = data_filt[data_filt['Mixing Tube Diameter'] == .030]
data_filt = data_filt[data_filt['Orifice'] < .015]
data_filt = data_filt[data_filt['Orifice'] > .011]

filt_size = data_filt.shape

index = 0;
sort_index = 0
for i in range (0,filt_size[0],1):
    if data_filt['Material'].iloc[index] == data_filt['Material'].iloc[i] and math.isclose(data_filt['Thickness'].iloc[index], data_filt['Thickness'].iloc[i], abs_tol= .02) == True and data_filt['Mixing Tube Diameter'].iloc[index] == data_filt['Mixing Tube Diameter'].iloc[i] and data_filt['Orifice'].iloc[index] == data_filt['Orifice'].iloc[i] and data_filt['Pressure'].iloc[index] == data_filt['Pressure'].iloc[i]:
        data_filt['Sorting'].iloc[i] = sort_index;
    else:
        index = i;
        sort_index = sort_index + 1
        data_filt['Sorting'].iloc[i] = sort_index;

size_y, size_x = data_filt.shape
psi_max = .69

save = [];
for i in range(0,size_y,1):
    save.append([psi_max,
                r_0(data_filt['Mixing Tube Diameter'].iloc[i]),
                hyd_pow(data_filt['Orifice'].iloc[i], data_filt['Pressure'].iloc[i]),
                data_filt['Actual Abrasive'].iloc[i] / water_flow_theo(data_filt['Orifice'].iloc[i], data_filt['Pressure'].iloc[i], 0),
                data_filt['Experimental Separation'].iloc[i],
                data_filt['Orifice'].iloc[i],
                data_filt['Thickness'].iloc[i]
                ])
    
save = np.array(save)

s_0m, s_0b, r_sm, r_sb = optimize_all([save[:,0],
                                       save[:,1],
                                       save[:,2],
                                       save[:,3],
                                       save[:,4],
                                       save[:,5],
                                       ])


r0 = save[:,1];
p_h = save[:,2];
r = save[:,3];
d_o = save[:,4];
mat_t = save[:,5];
mat = np.array(data_filt['Material'])

matm = 9.1
matb = -1.3

s_0 = s_0m*d_o + s_0b;
r_s = r_sm*d_o + r_sb;

v_sep = kin_equ(s_0m, s_0b, r_sm, r_sb, psi_max, r_0(data_filt['Mixing Tube Diameter']), p_h, r, d_o)
error =  v_sep/data_filt['Experimental Separation'] - 1


plt.figure()
plt.scatter(data_filt['Actual Abrasive'], data_filt['Experimental Separation'], label = 'experiemental')
plt.scatter(data_filt['Actual Abrasive'], v_sep, label = 'model')

plt.legend()
plt.figure()
plt.scatter(data_filt['Actual Abrasive'], error)

# %%
