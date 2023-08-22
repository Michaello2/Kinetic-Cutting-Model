
#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
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

def kin_equ(para, args):
    #factors:
    #mn = machineability number, l = material thickenss (in), mt = mixing tube diameter (in), ph = hydrualic powewr (hp),
    #r = abrasive loading, rs = shifting constant, s0 = scaling constant, r0 and psi_max = velocity scaling constant
    mn, l, mt, o, p, afr, psi_max, r0 = para
    s0, rs = args
    #calculate hydraulic power
    ph = hyd_pow(o,p)
    #machineability number relativized around aluminum
    #calc abrasive load ratio
    r = afr/water_flow_theo(o,p,0)
    #mute inputs:
    mt = 1
    l = 1;
    mn = mn/219

    v_sep = mn*l*mt*ph*s0*r/rs*(psi_max/(1+r/(rs*r0)))**2
    return v_sep

def sort_data(data):
    data_size = data.shape
    index = 0;
    sort_index = 0

    for i in range (0,data_size[0],1):
        data['Material'].iloc[i] = machine_num(data['Material'].iloc[i])   

        if data['Material'].iloc[index] == data['Material'].iloc[i] and math.isclose(data['Thickness'].iloc[index], data['Thickness'].iloc[i], abs_tol= .02) == True and data['Mixing Tube Diameter'].iloc[index] == data['Mixing Tube Diameter'].iloc[i] and data['Orifice'].iloc[index] == data['Orifice'].iloc[i] and data['Pressure'].iloc[index] == data['Pressure'].iloc[i]:
            data['Sorting'].iloc[i] = sort_index;
            
        else:
            index = i;
            sort_index = sort_index + 1
            data['Sorting'].iloc[i] = sort_index;
    return data

fname1 = 'separation_data.xlsx'
path1 = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + fname1;
data1 = pd.read_excel(io = path1, sheet_name = 'Sheet1')
data1.columns = ['Date',
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
data1 = data1.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
data1 = data1[data1['Mixing Tube Diameter'] == .03]
data1 = data1[data1['Orifice'] == .014]
data1 = data1[data1['Pressure'] > 50]
data1 = data1[data1['Material'] == 'Aluminum 6061']
data1 = data1[data1['Actual Abrasive']<5]
data1 = data1[data1['Experimental Separation']<500]
data1 = data1[data1['Thickness'] <1.1]
data1 = data1[data1['Thickness'] >.99]
#data1 = sort_data(data1)


fname2 = 'test_separation_data.xlsx'
path2 = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + fname2;
data2 = pd.read_excel(io = path2, sheet_name = 'Sheet1')
data2.columns = ['Date',
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
data2 = data2.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
data2 = data2[data2['Mixing Tube Diameter'] == .03]
data2 = data2[data2['Orifice'] == .014]
data2 = data2[data2['Pressure']> 55]
data2 = data2[data2['Material'] == 'Aluminum 6061']
data2 = data2[data2['Actual Abrasive'] < 5]
data2 = data2[data2['Experimental Separation']<500]
data2 = data2[data2['Thickness'] < 1.1]
data2 = data2[data2['Thickness'] > .99]
data2 = sort_data(data2)

mn = 219; l = 1; mt = .03; o = .014; p = 60; psi_max = .68;
afr = np.linspace(0,2.5,100)
para = mn, l, mt, o, p, afr, psi_max, r_0(mt)
para1 = mn, l, mt, o, p, data1['Actual Abrasive'], psi_max, r_0(mt)
para2 = mn, l, mt, o, p, data2['Actual Abrasive'], psi_max, r_0(mt)
s0 = 9
rs = .25
args = s0, rs

v_sep = kin_equ(para, args)
v_sep1 = kin_equ(para1, args)
v_sep2 = kin_equ(para2, args)


err1 = v_sep1/data1['Experimental Separation']-1
err2 = v_sep2/data2['Experimental Separation']-1

plt.figure()

plt.scatter(data1['Actual Abrasive'],data1['Experimental Separation'], color = 'blue', label = 'Measured database')
plt.scatter(data2['Actual Abrasive'],data2['Experimental Separation'], color = 'orange', label ='MAKE database')
plt.plot(afr,v_sep, label = 'KC Prediction')

plt.xlabel('Abrasive Flow Rate (lb/min)')
plt.ylabel('Separation Speed (ipm)')
plt.title('Separation Speed Vs Abrasive Feed Rate')
plt.legend()

plt.figure()
plt.scatter(data1['Actual Abrasive'],data1['Experimental Separation'], color = 'blue', label = 'Measured database')
plt.scatter(data2['Actual Abrasive'],data2['Experimental Separation'], color = 'orange', label ='MAKE database')
plt.plot(data1['Actual Abrasive'],v_sep1, linestyle = '--', marker='o',label = 'KC Model')
plt.xlabel('Abrasive Flow Rate (lb/min)')
plt.ylabel('Separation Speed (ipm)')
plt.title('KC Model Vs Measured Data')
plt.legend()

plt.figure()
plt.scatter(data1['Actual Abrasive'],err1, color = 'blue', label = 'Measured database')
plt.scatter(data2['Actual Abrasive'],err2, color = 'orange', label ='MAKE database')
plt.xlabel('Abrasive Flow Rate (lb/min)')
plt.ylabel('Error (Percent)')
plt.title('Model Error Vs Abrasive Feed Rate')
plt.legend()

plt.figure()
plt.hist(err2)
# %%
