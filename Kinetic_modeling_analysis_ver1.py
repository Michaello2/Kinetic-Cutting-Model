#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mp
import time
import pandas as pd
from functools import reduce
from collections import Counter
import Functions 


def func_rs(d_o):
    return -26.35*d_o + .69995

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

def func_S0(d_o,mat,mat_t):
    mn = machine_num(mat)
    return (-487.24*d_o+15.694)*(10.36*mat_t**-1.25)/(10.36*(1)**-1.25)*(mn/219)

def kin_equ(psi_max, r_0, p_h, r, r_s, s_0 ):
    v_sep = (p_h*s_0 /r_s)*r*((psi_max)/(1+(r/(r_0*r_s))))**2
    return v_sep

def opt_s0rs(para,vals):
    s_0 , r_s = para
    psi_max, r_0, p_h, r, v_sep = vals
    return np.sum(((p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def opt_rs(para,vals):
    r_s = para
    psi_max, r_0, s_0 , p_h, r, v_sep = vals
    return np.sum(((p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def opt_s0(para,vals):
    s_0 = para
    psi_max, r_0, r_s , p_h, r, v_sep = vals
    return np.sum(((p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def opt_all(para, vals):
    s_0m, s_0b, r_sm, r_sb = para;
    psi_max, r_0, p_h, r, v_sep, d_o = vals
    s_0 = s_0m*d_o + s_0b;
    r_s = r_sm*d_o + r_sb;
    return np.sum(((p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def optimize_all(vals):
    psi_max, r_0, p_h, r, v_sep, d_o = vals;
    guess = np.array([1,1,1,1])#np.array([-487.24, 15.69,-25.22,.698])
    inp = ([psi_max, r_0, p_h, r, v_sep, d_o])
    optim_all = sp.optimize.fmin(func = opt_all, x0 = guess, args = (inp,),  xtol = .0000001,maxiter = 500)
    return optim_all

def optimize_s0rs(vals):
    psi_max, r_0, p_h, r, v_sep = vals;
    guess = np.array([1,1])
    inp = ([psi_max, r_0, p_h, r, v_sep])
    optim_s0rs = sp.optimize.fmin(func = opt_s0rs, x0 = guess, args = (inp,))
    return optim_s0rs

def optimize_rs(vals):
    psi_max, r_0, s_0 , p_h, r, v_sep = vals;
    guess = 1
    inp = ([psi_max, r_0, s_0 , p_h, r, v_sep])
    optim_rs = sp.optimize.fmin(func = opt_rs, x0 = guess, args = (inp,))
    return optim_rs

def optimize_s0(vals):
    psi_max, r_0, r_s , p_h, r, v_sep = vals;
    guess = 1
    inp = ([psi_max, r_0, r_s , p_h, r, v_sep])
    optim_s0 = sp.optimize.fmin(func = opt_s0, x0 = guess, args = (inp,))
    return optim_s0

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

def velmod_const (mt):
    psi_max = {
        '0.03': .696,
        '0.036': .707,
        '0.042': .661,
        '0.048': .684
    }

    r_0 = {
        '0.03': .632,
        '0.036': .799,
        '0.042': 1.08,
        '0.048': 1.03
    }
    
    if str(mt) in psi_max:
        output = [psi_max[str(mt),r_0[str(mt)]]]
    else:
        #if not within the given range, use the average psi_max since it is constant over mixingtube diamter:
        #use a linear function that describes the change in r_0 over mixing tube diamter.
        output = [.684, 24.583*(float(mt))-.0735]
    return output


file_name = 'test_separation.xlsx'
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

ind = data_sep.loc[data_sep['Sorting'].idxmax()]
r = np.linspace(0,1,100)

#kinetic model constants
s0 = [];
rs = [];
rs_const_s0 = [];
s0_func_rs = [];
s0_calc = [];

#parameters
lab_save = [];
hp_save = [];
mt_save = [];
do_save = [];
p_save = [];
mat_save = [];
mat_t_save = [];
alr_save = [];

#error savs
err_s0rs = [];
err_rs = [];
err_s0 = [];
err_const = []; 
err_all = [];

psi_max = {
    '0.03': .696,
    '0.036': .707,
    '0.042': .661,
    '0.048': .684
    }
r_0 = {
    '0.03': .632,
    '0.036': .799,
    '0.042': 1.08,
    '0.048': 1.03
    }

#Linear optimization of the entiredata set.
data_filt = data_sep[data_sep['Material'] == 'Aluminum 6061']
# data_filt = data_filt[data_filt['Thickness'] == 1]
data_filt = data_filt[data_filt['Mixing Tube Diameter'] == .03]
# data_filt = data_filt[data_filt['Orifice'] == .01]
#data_filt = data_filt[data_filt['Pressure'] == 60]
data_filt = data_filt[data_filt['Actual Abrasive'] < 3]
data_filt = data_filt[data_filt['Experimental Separation'] < 500]

counts = [Counter(data_filt['Mixing Tube Diameter']).keys(),
          Counter(data_filt['Orifice']).keys(),
          Counter(data_filt['Material']).keys(),
          Counter(data_filt['Pressure']).keys(),
          Counter(data_filt['Thickness']).keys(),
          ]



size_y, size_x = data_filt.shape
save = [];
for i in range(0,size_y,1):
    save.append([psi_max[str(data_filt['Mixing Tube Diameter'].iloc[i])],
                r_0[str(data_filt['Mixing Tube Diameter'].iloc[i])],
                hyd_pow(data_filt['Orifice'].iloc[i],data_filt['Pressure'].iloc[i]),
                data_filt['Actual Abrasive'].iloc[i]/water_flow_meas(data_filt['Orifice'].iloc[i], data_filt['Pressure'].iloc[i]),
                data_filt['Experimental Separation'].iloc[i],
                data_filt['Orifice'].iloc[i]])
    
save = np.array(save)

s_0m, s_0b, r_sm, r_sb = optimize_all([save[:,0], save[:,1], save[:,2], save[:,3], save[:,4], save[:,5]])

err_ind = (save[:, 4] / kin_equ(save[:,0],save[:,1],save[:,2],save[:,3],r_sm*save[:,5]+r_sb,s_0m*save[:,5]+s_0b))-1

ax = plt.subplot()
ax.scatter(save[:,3],save[:,4])
ax.set_title('Error Vs Abrasive Feed Rate')
ax.set_xlabel('Abrasive Loading (R)')
ax.set_ylabel('Error')
#ax.yaxis.set_major_formatter(mp.ticker.PercentFormatter())

print('S_0 Slope: {0}, S_0 Intercept: {1}, R_s slope: {2}, R_s Intercept: {3}'.format(s_0m, s_0b, r_sm, r_sb))


plt.figure()
plt.title('Error vs Hydraulic Power')
plt.scatter(np.array(p_save),np.array(err_all))
plt.xlabel('hydraulic power (hp)')
plt.ylabel('Error')

#save to excel
file_name = 'Kinetic Cutting Model Results'
file_dir = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code'
file_path = file_dir + '/' + file_name;
if '.xlsx' not in file_name:
    file_path = file_path + '.xlsx';

res = [lab_save, mat_save, mat_t_save, mt_save, do_save, p_save, hp_save, s0, rs, rs_const_s0, s0_func_rs, err_s0rs, err_rs, err_s0, err_const]
results_save = pd.DataFrame(res).T
results_save.columns = ['Label', 'Material Type', 'Material Thickness (in)','Mixing Tube Diameter (in)','Orifice Diameter (in)','Pressure (ksi)', 'Hydraulic Power (hp)', 'S_0', 'R_s', 'R_s with constant S_0', 'S_0 with r_s Function', 'Error', 'Error with Constant S_0', 'Error with R_s Function', 'Error with Constant S_0 and R_s Function']

res1 = [s_0m, s_0b, r_sm, r_sb]
results_save1 = pd.DataFrame(res1).T
results_save1.columns = ['S_0 slope','S_0 intercept','R_s slope','R_s intercept']

writer = pd.ExcelWriter(file_path, engine = "xlsxwriter")
results_save.to_excel(writer, sheet_name = 'Parameters')
results_save1.to_excel(writer, sheet_name = 'Mass Optimization')

workbook = writer.book

writer.close()



# %%
