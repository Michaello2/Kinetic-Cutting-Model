#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import pandas as pd

def func_rs(p_h):
    return 0.0003*p_h**2 - 0.0206*p_h + 0.6366

def kin_equ(psi_max, r_0, p_h, r, r_s, r_s ):
    v_sep = (p_h*r_s /r_s)*r*((psi_max)/(1+(r/(r_0*r_s))))**2
    return v_sep

def opt_s0_rs(para,vals):
    r_s , r_s = para
    psi_max, r_0, p_h, r, v_sep = vals
    return np.sum(((p_h*r_s /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def opt_rs(para,vals):
    r_s = para
    psi_max, r_0, s_0 , p_h, r, v_sep = vals
    return np.sum(((p_h*s_0 /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def opt_s0(para,vals):
    s_0 = para
    psi_max, r_0, r_s , p_h, r, v_sep = vals
    return np.sum(((p_h*r_s /r_s*r*((psi_max)/(1+(r/(r_0*r_s))))**2)/v_sep - 1)**2)

def optimize_s0_Rs(vals):
    psi_max, r_0, p_h, r, v_sep = vals;
    guess = np.array([1,1])
    inp = ([psi_max, r_0, p_h, r, v_sep])
    optim_s0rs = sp.optimize.fmin(func = opt_s0_rs, x0 = guess, args = (inp,))
    return optim_s0rs

def optimize_Rs(vals):
    psi_max, r_0, s_0 , p_h, r, v_sep = vals;
    guess = 1
    inp = ([psi_max, r_0, s_0 , p_h, r, v_sep])
    optim_rs = sp.optimize.fmin(func = opt_rs, x0 = guess, args = (inp,))
    return optim_rs

def optimize_s0(vals):
    psi_max, r_0, s_0 , p_h, r, v_sep = vals;
    guess = 1
    inp = ([psi_max, r_0, s_0 , p_h, r, v_sep])
    optim_s0rs = sp.optimize.fmin(func = opt_rs, x0 = guess, args = (inp,))
    return optim_s0rs


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

file_name = 'separation_data.xlsx'
my_path = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + file_name;
data_sep = pd.read_excel(io = my_path, sheet_name = 'Sheet1')
data_sep.columns = ['Date', 'Sorting', 'Sort', 'Material',  'Thickness', 'Table', 'Machine', 'Pump', 'Nozzle', 'Abrasive', 'On/Off Valve', 'On/Off Valve Type', 'Cutting Head Number', 'Orifice', 'Mixing Tube Diameter', 'Mixing Tube Length', 'Flow Conditioner', 'Pressure', 'Water Flow Rate (gpm)', 'Desired Abrasive Feed Rate', 'Actual Abrasive', 'Material Separation Speed', 'Percentage 1', 'Percentage 2', 'Percentage 3', 'Experimental Separation', 'Abrasive Loading']             

ind = data_sep.loc[data_sep['Sorting'].idxmax()]
r = np.linspace(0,1,100)

s0_save = [];
rs_save = [];
rs_save_con = [];
lab_save = [];
hp_save = [];
mt_save = [];
do_save = [];
p_save = [];
mat_save = [];
mat_t_save = [];
err_s0rs = [];
err_rs = [];
err_const = []; 
s0_const = 10.8

v_sep_consant = []

plt.figure()
for i in range(1,ind['Sorting']+1,1):
    #separate each test
    data  = data_sep.loc[data_sep["Sorting"] == i]
    data.reset_index(drop=True, inplace=True)

    #set parameters 
    d_o = data['Orifice'].iloc[0]   #orifice diamter 
    p = data["Pressure"].iloc[0]    #Pressure
    mt = data['Mixing Tube Diameter'].iloc[0]   #mixing tube diameter
    afr = data['Actual Abrasive'].values    #abrasive feed rate
    sep = data['Experimental Separation'].values    #separation speed

    #calcuated parameters
    wfr = water_flow_meas(d_o,p)    #Water flow rate
    p_h = hyd_pow(d_o,p)    #hydraulic power
    alr = afr/wfr   #Find abrasive loading

#===================================================================================================   
    #optimize scaling constant S_0 and R_s for the test. This functions optimizes using both parameters.
    optim_s0rs = optimize_s0_Rs([psi_max[str(mt)], r_0[str(mt)], p_h, alr, sep])
    #find kin model separaiton speeds with optimized S_0 and R_s
    v_sepmod_s0rs = kin_equ(psi_max[str(mt)], r_0[str(mt)], p_h, r, optim_s0rs[1], optim_s0rs[0])
    #calculate error between model and measured data.
    error_s0rs = (np.sum(kin_equ(psi_max[str(mt)],r_0[str(mt)],p_h, alr,optim_s0rs[1],optim_s0rs[0])/sep-1))**2
#===================================================================================================
    #optimize scaling constant R_s while keeping S_0 constant.
    optim_rs = optimize_Rs([psi_max[str(mt)], r_0[str(mt)], s0_const, p_h, alr, sep])
    #find kin model separation speeds with optimized R_s
    v_sepmod_rs = kin_equ(psi_max[str(mt)], r_0[str(mt)], p_h, r, optim_rs[0], s0_const)
    #calculated error between model and measured data.
    error_rs = (np.sum(kin_equ(psi_max[str(mt)],r_0[str(mt)],p_h, alr,optim_rs[0],s0_const)/sep-1))**2
#===================================================================================================
    #Calculated separation speed using constant S_0 and a R_s function
    v_sepmod_const = kin_equ(psi_max[str(mt)],r_0[str(mt)], p_h, r, func_rs(p_h), s0_const)
    print(func_rs(p_h))
    #calculate error
    error_const = 1-(np.sum(kin_equ(psi_max[str(mt)],r_0[str(mt)], p_h, alr, func_rs(p_h), s0_const)/sep-1))**2
#===================================================================================================
    #calcuated separation speed uing R_s function optimiizing r_s 



    lab = 'Label: {0}, {1}, {2}, {3}, {4}, {5}'.format(data["Material"].iloc[0],data["Thickness"].iloc[0],data["Nozzle"].iloc[0], data["Pressure"].iloc[0],data["Orifice"].iloc[0],data["Mixing Tube Diameter"].iloc[0])
    
    lab_save.append(lab)
    s0_save.append(float(optim_s0rs[0]))
    rs_save.append(float(optim_s0rs[1]))
    hp_save.append(float(p_h))
    mt_save.append(float(mt))
    do_save.append(float(d_o))
    p_save.append(p)
    mat_save.append(str(data['Material'].iloc[0]))
    mat_t_save.append(str(data['Thickness'].iloc[0]))

    err_s0rs.append(error_s0rs)
    err_rs.append(error_rs)
    err_const.append(error_const)
    rs_save_con.append(optim_rs[0])
    
    print(mt)
    if str(mt) == '0.03' and str(p) == '60':
        plt.scatter(alr,sep, label = lab)
        plt.plot(r,v_sepmod_const)
plt.xlabel('Abrasive Loading')
plt.ylabel( 'Separation Speed (ipm)')
plt.legend()

plt.figure()
plt.scatter(np.array(hp_save), np.array(s0_save))
plt.xlabel('Hydraulic Power')
plt.ylabel('Scaling')

plt.figure()
plt.scatter(np.array(hp_save), np.array(rs_save))
plt.xlabel('Hydraulic Power')
plt.ylabel('R_s')

plt.figure()
plt.scatter(np.array(hp_save),np.array(err_const))
plt.xlabel('hydraulic power (hp)')
plt.ylabel('Error')

file_name = 'Kinetic Cutting Model Results'
file_dir = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code'
file_path = file_dir + '/' + file_name;
if '.xlsx' not in file_name:
    file_path = file_path + '.xlsx';

res = [lab_save, mat_save, mat_t_save, mt_save, do_save, p_save, s0_save, rs_save, hp_save, rs_save_con, err_s0rs, err_rs]
results_save = pd.DataFrame(res).T
results_save.columns = ['Label', 'Material Type', 'Material Thickness (in)','Mixing Tube Diameter (in)','Orifice Diameter (in)','Pressure (ksi)','S_0', 'R_s', 'Hydraulic Power (hp)', 'R_s with constant S_0','Error', 'Error Constant']
writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
results_save.to_excel(writer, sheet_name = 'Parameters')
workbook = writer.book

writer.close()


r = np.linspace(0,1,1000);
psi = psi_max['0.03']
r_zero = r_0['0.03']
plt.figure()
plt.plot(r,kin_equ(psi,r_zero,30,r,.4,10),label = 'baseline',color ='blue')
plt.plot(r,kin_equ(psi,r_zero,30,r,.5,10),label = 'r_s change',color ='orange')
plt.plot(r,kin_equ(psi,r_zero,30,r,.5,12),label = 'r_s  change',color ='orange')
plt.legend()

# %%
