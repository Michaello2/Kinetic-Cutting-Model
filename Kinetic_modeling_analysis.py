#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mp
import time
import pandas as pd
from functools import reduce
from collections import Counter
import math

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

def psi_max (mt):
    return .687

def r_0 (mt):
    return 24.58*mt - .0735; 


#Linear optimization of the entiredata set.
data_filt = data_sep[data_sep['Material'] == 'Aluminum 6061']
# data_filt = data_filt[data_filt['Thickness'] == 1]
#data_filt = data_filt[data_filt['Mixing Tube Diameter'] == .036]
# data_filt = data_filt[data_filt['Orifice'] == .01]
#data_filt = data_filt[data_filt['Pressure'] == 60]
data_filt = data_filt[data_filt['Actual Abrasive'] < 3]
data_filt = data_filt[data_filt['Experimental Separation'] < 500]

mt_count = Counter(data_filt['Mixing Tube Diameter']).keys()

size_y, size_x = data_filt.shape
save = [];
for i in range(0,size_y,1):
    save.append([psi_max(data_filt['Mixing Tube Diameter'].iloc[i]),
                r_0(data_filt['Mixing Tube Diameter'].iloc[i]),
                hyd_pow(data_filt['Orifice'].iloc[i],data_filt['Pressure'].iloc[i]),
                data_filt['Actual Abrasive'].iloc[i]/water_flow_meas(data_filt['Orifice'].iloc[i], data_filt['Pressure'].iloc[i]),
                data_filt['Experimental Separation'].iloc[i],
                data_filt['Orifice'].iloc[i]])
    
save = np.array(save)

s_0m, s_0b, r_sm, r_sb = optimize_all([save[:,0], save[:,1], save[:,2], save[:,3], save[:,4], save[:,5]])

err_ind = (save[:, 4] / kin_equ(save[:,0],save[:,1],save[:,2],save[:,3],r_sm*save[:,5]+r_sb,s_0m*save[:,5]+s_0b))-1

ax = plt.subplot()
ax.scatter(save[:,3],err_ind*100)
ax.set_title('Error Vs Abrasive Feed Rate')
ax.set_xlabel('Abrasive Loading (R)')
ax.set_ylabel('Error')
ax.yaxis.set_major_formatter(mp.ticker.PercentFormatter())

data_sort = data_filt.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
sort_size = data_sort.shape

index = 0;
sort_index = 0
for i in range (0,sort_size[0],1):
    if data_sort['Material'].iloc[index] == data_sort['Material'].iloc[i] and math.isclose(data_sort['Thickness'].iloc[index], data_sort['Thickness'].iloc[i], abs_tol= .02) == True and data_sort['Mixing Tube Diameter'].iloc[index] == data_sort['Mixing Tube Diameter'].iloc[i] and data_sort['Orifice'].iloc[index] == data_sort['Orifice'].iloc[i] and data_sort['Pressure'].iloc[index] == data_sort['Pressure'].iloc[i]:
        data_sort['Sorting'].iloc[i] = sort_index;
    else:
        index = i;
        sort_index = sort_index + 1
        data_sort['Sorting'].iloc[i] = sort_index;


ind = data_sort.loc[data_sort['Sorting'].idxmax()]

plt.figure()
for i in range(1,ind['Sorting']+1,1):
    #separate each test
    data  = data_sort.loc[data_sort["Sorting"] == i]
    data_size = data.shape
    if data_size[0] > 2:
        data.reset_index(drop=True, inplace=True)

        #set parameters 
        d_o = data['Orifice'].iloc[0]   #orifice diamter 
        p = data["Pressure"].iloc[0]    #Pressure
        mt = float(data['Mixing Tube Diameter'].iloc[0])   #mixing tube diameter
        mat = data['Material'].iloc[0]  #material type
        mat_t = data['Thickness'].iloc[0] #material thickness
        afr = data['Actual Abrasive'].values    #abrasive feed rate
        sep = data['Experimental Separation'].values    #separation speed

        #calcuated parameters
        wfr = water_flow_meas(d_o,p)    #Water flow rate
        p_h = hyd_pow(d_o,p)    #hydraulic power
        alr = afr/wfr   #Find abrasive loading

    #===================================================================================================   
        #optimize scaling constant S_0 and R_s for the test. This functions optimizes using both parameters.
        optim_s0rs = optimize_s0rs([psi_max(mt), r_0(mt), p_h, alr, sep])
        #find kin model separaiton speeds with optimized S_0 and R_s
        v_sepmod_s0rs = kin_equ(psi_max(mt), r_0(mt), p_h, r, optim_s0rs[1], optim_s0rs[0])
        #calculate error between model and measured data.
        error_s0rs = np.abs(np.sum(kin_equ(psi_max(mt),r_0(mt),p_h, alr,optim_s0rs[1],optim_s0rs[0])/sep-1))/len(alr)
    #===================================================================================================
        s0_const = func_S0(d_o,mat,mat_t)
        #optimize scaling constant R_s while keeping S_0 constant.
        optim_rs = optimize_rs([psi_max(mt), r_0(mt), s0_const, p_h, alr, sep])
        #find kin model separation speeds with optimized R_s
        v_sepmod_rs = kin_equ(psi_max(mt), r_0(mt), p_h, r, optim_rs[0], s0_const)
        #calculated error between model and measured data.
        error_rs = np.abs(np.sum(kin_equ(psi_max(mt),r_0(mt),p_h, alr,optim_rs[0],s0_const)/sep-1))/len(alr)
    #===================================================================================================
        rs_const = func_rs(d_o)
        optim_s0 =  optimize_s0([psi_max(mt), r_0(mt), rs_const, p_h, alr, sep])
        #calcuated separation speed using R_s function optimiizing s_0 
        v_sepmod_s0 = kin_equ(psi_max(mt),r_0(mt), p_h, r, rs_const, optim_s0[0])
        error_s0 = np.abs(np.sum(kin_equ(psi_max(mt),r_0(mt),p_h, alr,rs_const,optim_s0[0])/sep-1))/len(alr)
    #===================================================================================================
        #Calculated separation speed using constant S_0 and a R_s function
        v_sepmod_const = kin_equ(psi_max(mt),r_0(mt), p_h, r, rs_const, s0_const)
        #calculate error
        error_const = np.abs(np.sum(kin_equ(psi_max(mt),r_0(mt), p_h, alr, rs_const, s0_const)/sep-1))/len(alr)
    #===================================================================================================
        #Calculate separation speed using mass calculation of S_0 and R_s
        v_sepmod_all = kin_equ(psi_max(mt), r_0(mt), p_h, r, r_sm*d_o+r_sb, s_0m*d_o+s_0b)
        #calcualted error between mass calculation and measured data.
        error_all = np.abs(np.sum(kin_equ(psi_max(mt), r_0(mt), p_h, alr, r_sm*d_o+r_sb, s_0m*d_o+s_0b)/sep-1))/len(alr)

        #label data set
        lab = 'Label: {0}, {1}, {2}, {3}, {4}, {5}'.format(data["Material"].iloc[0],data["Thickness"].iloc[0],data["Nozzle"].iloc[0], data["Pressure"].iloc[0],data["Orifice"].iloc[0],data["Mixing Tube Diameter"].iloc[0])
        
        go = 1
        if go == 1: #
            #save parameters:
            print(d_o)
            s0_calc.append(s0_const)
            lab_save.append(lab)
            hp_save.append(float(p_h))
            mt_save.append(float(mt))
            do_save.append(float(d_o))
            p_save.append(p)
            mat_save.append(str(data['Material'].iloc[0]))
            mat_t_save.append(str(data['Thickness'].iloc[0]))
            alr_save.append(alr)

            #save model constants:
            s0.append(float(optim_s0rs[0]))
            rs.append(float(optim_s0rs[1]))
            rs_const_s0.append(optim_rs[0])
            s0_func_rs.append(optim_s0[0])
            
            #save error 
            err_s0rs.append(error_s0rs)
            err_rs.append(error_rs)
            err_s0.append(error_s0)
            err_const.append(error_const)
            err_all.append(error_all)

            plt.scatter(alr,sep, label = lab)
            plt.plot(r,v_sepmod_s0rs)
plt.title('Model vs Measured')
plt.xlabel('Abrasive Loading')
plt.ylabel( 'Separation Speed (ipm)')
# plt.legend()   

# print('S_0 Slope: {0}, S_0 Intercept: {1}, R_s slope: {2}, R_s Intercept: {3}'.format(s_0m, s_0b, r_sm, r_sb))

# plt.figure()
# plt.title('Scaling Vs Hydraulic Power')
# #plt.scatter(np.array(hp_save), np.array(s0))
# plt.scatter(np.array(do_save),np.array(rs_const_s0))
# plt.plot(o,s_0)
# # plt.scatter(np.array(p_save),np.array(s0_calc))
# plt.xlabel('Pressure (ksi)')
# plt.ylabel('Scaling')

# plt.figure()
# plt.title('Shifting vs Hydraulic Power')
# plt.scatter(np.array(p_save), np.array(rs))
# #plt.scatter(np.array(hp_save), np.array(rs_const_s0))
# plt.xlabel('Pressure (ksi)')
# plt.ylabel('R_s')

plt.figure()
plt.title('Error vs Hydraulic Power')
plt.scatter(np.array(p_save),np.array(err_s0rs))
plt.xlabel('hydraulic power (hp)')
plt.ylabel('Error')

# plt.figure()
# plt.title('Error Histogram')
# n, bins, patches = plt.hist(x=np.array(err_s0rs), bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)



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





# r = np.linspace(0,1,1000);
# psi = psi_max['0.03']
# r_zero = r_0['0.03']
# plt.figure()
# plt.plot(r,kin_equ(psi,r_zero,30,r,.4,10),label = 'baseline',color ='blue')
# plt.plot(r,kin_equ(psi,r_zero,30,r,.5,10),label = 'r_s change',color ='orange')
# plt.plot(r,kin_equ(psi,r_zero,30,r,.5,12),label = 'r_s  change',color ='orange')
# plt.legend()

# %%
