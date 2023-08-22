#%%

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
from collections import Counter
import math
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

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

def kin_equ_rs(para, args):
    #factors:
    #mn = machineability number, l = material thickenss (in), mt = mixing tube diameter (in), ph = hydrualic powewr (hp),
    #r = abrasive loading, rs = shifting constant, s0 = scaling constant, r0 and psi_max = velocity scaling constant
    mn, l, mt, o, p, afr, psi_max, r0, rs = para
    s0 = args
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

def s0rs_opt(args, para, v_sep):
    v_sepmod = kin_equ(para, args)
    error = np.sum((v_sepmod/v_sep - 1)**2)
    return error

def rs_opt(args, para, v_sep):
    v_sepmod = kin_equ_rs(para, args)
    error = np.sum((v_sepmod/v_sep - 1)**2)
    return error  

def reg_s0rs(vals, v_sep):
    mat, l, mt, o, p, afr, psi_max, r0 = vals;
    guess = np.array([1,1])
    inp = ([mat, l, mt, o, p, afr, psi_max, r0])
    optim_all = sp.optimize.fmin(func = s0rs_opt, x0 = guess, args = (inp, v_sep,))
    return optim_all

def reg_rs(vals,v_sep): 
    mat, l, mt, o, p, afr, psi_max, r0 = vals;
    guess = np.array([1])
    rs = r_s(p);
    inp = ([mat, l, mt, o, p, afr, psi_max, r0, rs])
    optim_all = sp.optimize.fmin(func = rs_opt, x0 = guess, args = (inp, v_sep,))
    return optim_all, rs

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

def ind_reg(data):
    data_size = data.shape
    save = []
    index = 0;
    sort_index = 0

    for i in range (0,data_size[0],1):
        if data['Material'].iloc[index] == data['Material'].iloc[i] and math.isclose(data['Thickness'].iloc[index], data['Thickness'].iloc[i], abs_tol= .02) == True and data['Mixing Tube Diameter'].iloc[index] == data['Mixing Tube Diameter'].iloc[i] and data['Orifice'].iloc[index] == data['Orifice'].iloc[i] and data['Pressure'].iloc[index] == data['Pressure'].iloc[i]:
            data['Sorting'].iloc[i] = sort_index;
            
        else:
            index = i;
            sort_index = sort_index + 1
            data['Sorting'].iloc[i] = sort_index;
    
    for i in range(0,sort_index,1):
        section  = data.loc[data["Sorting"] == i]
        data_size = section.shape
        if data_size[0] > 2:
            #Material properties and cutting parameters.
            mat = section['Material'].iloc[0];
            l = section['Thickness'];
            mt = section['Mixing Tube Diameter'];
            o = section['Orifice'];
            p = section['Pressure'];
            afr = section["Actual Abrasive"];
            psi_max = .68;
            r0 = .58;
            para = [mat, l, mt, o, p, afr, psi_max, r0]

            #Modeling values
            v_sep = section['Experimental Separation']
            s0, rs = reg_s0rs(para, v_sep)

            save.append([s0, rs, mat, l.iloc[0], mt.iloc[0], o.iloc[0], p.iloc[0], psi_max, r0])

    return save

def ind_reg_rs(data):
    data_size = data.shape
    save = []
    index = 0;
    sort_index = 0

    for i in range (0,data_size[0],1):
        if data['Material'].iloc[index] == data['Material'].iloc[i] and math.isclose(data['Thickness'].iloc[index], data['Thickness'].iloc[i], abs_tol= .02) == True and data['Mixing Tube Diameter'].iloc[index] == data['Mixing Tube Diameter'].iloc[i] and data['Orifice'].iloc[index] == data['Orifice'].iloc[i] and data['Pressure'].iloc[index] == data['Pressure'].iloc[i]:
            data['Sorting'].iloc[i] = sort_index;
            
        else:
            index = i;
            sort_index = sort_index + 1
            data['Sorting'].iloc[i] = sort_index;
    
    for i in range(0,sort_index,1):
        section  = data.loc[data["Sorting"] == i]
        data_size = section.shape
        if data_size[0] > 1:
            #Material properties and cutting parameters.
            mat = section['Material'].iloc[0];
            l = section['Thickness'];
            mt = section['Mixing Tube Diameter'];
            o = section['Orifice'];
            p = section['Pressure'];
            afr = section["Actual Abrasive"];
            psi_max = .68;
            r0 = .58;
            para = [mat, l, mt, o, p, afr, psi_max, r0]

            #Modeling values
            v_sep = section['Experimental Separation']
            s0, rs = reg_rs(para, v_sep)
            rs = np.array(rs)

            save.append([s0[0], rs[0], mat, l.iloc[0], mt.iloc[0], o.iloc[0], p.iloc[0], psi_max, r0])

    return save

def r_s(p):
    return .00588 * p - .0734

def s_0(o):
    return -1246.27 * o + 27.034

#=======================================================================================
#import training set
train_fname = 'separation_data.xlsx'
train_path = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + train_fname;
train_data = pd.read_excel(io = train_path, sheet_name = 'Sheet1')

train_data.columns = ['Date',
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
train_data = train_data.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
train_data = train_data[train_data['Experimental Separation'] < 100]
train_data_filt = train_data[train_data['Mixing Tube Diameter'] == .03]
train_data_filt = train_data_filt[train_data_filt['Material'] == 'Aluminum 6061']
train_data_filt = train_data_filt[train_data_filt['Thickness'] == 1]
#sort data into by material type, material thickness, mixing tube diameter, orifice diameter, pressure, and abrasive feed rate.
train_data_filt = sort_data(train_data_filt)

#=======================================================================================
#import test set
test_fname = 'make_separation_data.xlsx'
test_path = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + test_fname;
test_data = pd.read_excel(io = test_path, sheet_name = 'Sheet1')

test_data.columns = ['Date',
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
test_data = test_data.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
test_data_filt = test_data[test_data['Mixing Tube Diameter'] == .03]
test_data_filt = test_data_filt[test_data_filt['Material'] == 'Aluminum 6061']
test_data_filt = test_data_filt[test_data_filt['Thickness'] == 1]
#sort data into by material type, material thickness, mixing tube diameter, orifice diameter, pressure, and abrasive feed rate.
test_data_filt = sort_data(test_data_filt)

#=======================================================================================
const_fname = 'test_separation_data.xlsx'
const_path = r'J:\Engineering\private\_Staff\Michael Lo\VS Code\Python Code' + '\\' + const_fname;
const_data = pd.read_excel(io = const_path, sheet_name = 'Sheet1')

const_data.columns = ['Date',
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
const_data = const_data.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
const_data_filt = const_data[const_data['Mixing Tube Diameter'] == .03]
const_data_filt = const_data_filt[const_data_filt['Material'] == 'Aluminum 6061']
const_data_filt = const_data_filt[const_data_filt['Thickness'] > .9]
const_data_filt = const_data_filt[const_data_filt['Thickness'] < 1.1]
const_data_filt = const_data_filt[const_data_filt['Experimental Separation'] < 500]
#sort data into by material type, material thickness, mixing tube diameter, orifice diameter, pressure, and abrasive feed rate.
const_data_filt = sort_data(const_data_filt)

#=======================================================================================
#Training data
train_return = np.array(ind_reg(train_data_filt))
s0_train = train_return[:,0]; rs_train = train_return[:,1]; mat_train = train_return[:,2]; l_train = train_return[:,3];
mt_train = train_return[:,4]; o_train = train_return[:,5]; p_train = train_return[:,6]; psi_max_train = train_return[:,7]; r0_train = train_return[:,8]

train_ind = np.array([mat_train, l_train, mt_train, o_train, p_train, psi_max_train, r0_train]).T
train_dep = np.array([s0_train, rs_train]).T

train_model_lin = LinearRegression()
train_model_tree = DecisionTreeRegressor()
train_model_lin.fit(train_ind,train_dep)
train_model_tree.fit(train_ind,train_dep)

#=======================================================================================
#Testing data
test_return = np.array(ind_reg(test_data_filt))
s0_test = test_return[:,0]; rs_test = test_return[:,1]; mat_test = test_return[:,2]; l_test = test_return[:,3];
mt_test = test_return[:,4]; o_test = test_return[:,5]; p_test = test_return[:,6]; psi_max_test = test_return[:,7]; r0_test = test_return[:,8]

test_ind = np.array([mat_test, l_test, mt_test, o_test, p_test, psi_max_test, r0_test]).T
test_dep = np.array([s0_test, rs_test]).T

test_model_lin = LinearRegression()
test_model_tree = DecisionTreeRegressor()
test_model_lin.fit(test_ind,test_dep)
test_model_tree.fit(test_ind,test_dep)

#=======================================================================================
test_return_rs = np.array(ind_reg(const_data_filt)) #np.array(ind_reg_rs(const_data_filt), dtype = object)
s0_test_rs = test_return_rs[:,0]; rs_test_rs = test_return_rs[:,1]; mat_test_rs = test_return_rs[:,2]
l_test_rs = test_return_rs[:,3]; mt_test_rs = test_return_rs[:,4]; o_test_rs = test_return_rs[:,5]
p_test_rs = test_return_rs[:,6]; psi_test_rs = test_return_rs[:,7]; r0_test_rs = test_return_rs[:,8]

test_ind_rs = np.array([mat_test_rs, l_test_rs, mt_test_rs, o_test_rs, p_test_rs, psi_test_rs, r0_test_rs],dtype = float).T
test_dep_rs = np.array([s0_test_rs, rs_test_rs]).T

test_model_rs = LinearRegression()
test_model_rs.fit(test_ind_rs,test_dep_rs)
#=======================================================================================
#Test Error:



err_test_lin = test_model_lin.predict(test_ind)

plt.figure()
plt.title('Individual Error Between Model and Measured Data (Train = Measured)')
plt.xlabel('Orifice Diameter (in)')
plt.ylabel('Individual Error (Percent)')
plt.scatter(o_test,err_test_lin[:,0])

plt.figure()
plt.title('Individual Error Between Model and Measured Data (Train = Measured)')
plt.xlabel('Pressure (ksi)')
plt.ylabel('Individual Error (Percent)')
plt.scatter(p_test,err_test_lin[:,0])


#=======================================================================================
l = 1;
mat = 'Aluminum 6061'
o = np.linspace(.01,.020,20)
mt = .03
p = np.linspace(40,85,20)
psi_max = .68;
r0 = .57;

#=======================================================================================
ax = plt.figure().add_subplot(projection = '3d')
prog_test = np.zeros(shape = (20,20))
prog_train = np.zeros(shape = (20,20))
prog_const = np.zeros(shape = (20,20))

plt.figure()
for i in range(0,20,1):
    for j in range(0,20,1):
        mod_input = [machine_num(mat), l, mt, o[i], p[j], psi_max, r0]

        test_pred_lin = test_model_lin.predict([mod_input])
        test_pred_tree = test_model_tree.predict([mod_input])

        test_pred_rs = test_model_rs.predict([mod_input])

        train_pred_lin = train_model_lin.predict([mod_input])      
        train_pred_tree = train_model_tree.predict([mod_input])
        
        test_s0_rs = float(test_pred_rs[0,0])
        test_rs_rs = float(test_pred_rs[0,1])
        test_s0_lin = float(test_pred_lin[0,0])
        test_rs_lin = float(test_pred_lin[0,1])
        test_s0_tree = float(test_pred_tree[0,0])
        test_rs_tree = float(test_pred_tree[0,1])
        train_s0_lin = float(train_pred_lin[0,0])
        train_rs_lin = float(train_pred_lin[0,1])
        train_s0_tree = float(train_pred_tree[0,0])
        train_rs_tree = float(train_pred_tree[0,1])


        prog_test[j,i] = test_s0_lin
        prog_train[j,i] = train_s0_lin
        prog_const[j,i] = test_s0_rs



        plt.scatter(o[i],(test_s0_lin), color = 'blue')
        # plt.scatter(o[i],(train_s0_lin), color = 'orange')
        # plt.scatter(o[i],(test_s0_rs), color =  'green')
        


o, p = np.meshgrid(o,p)
#ax.plot_surface(o,p,prog_const, color = 'blue', label = 'test set 2', alpha = .2)
# ax.plot_surface(o,p,prog_test, color = 'green', label = 'test set 1')
ax.plot_surface(o,p,prog_train, color = 'orange', label = 'training set', alpha = .2)


test_return = np.array(test_return)
# ax.scatter(test_return[:,5], test_return[:,6], test_return[:,0])
ax.scatter(o_train, p_train, s0_train, color = 'black', label = 'Raw Train Data')
# ax.scatter(o_test, p_test, s0_test, color = 'blue', label = 'Raw Test Data')
ax.set_xlabel('Orifice Diameter (in)')
ax.set_ylabel('Pressure (ksi)')
ax.set_zlabel('Scaling Constant S0')
#ax.legend()



s0_m, s0_b = np.polyfit(o_test,s0_test, deg = 1)
y = s0_m * o_test + s0_b
plt.figure()
plt.subplot(211)
plt.scatter(o_train, s0_train)
plt.scatter(o_test, s0_test)
#plt.scatter(o_test_rs, s0_test_rs)
plt.plot(o_test, y)
plt.xlabel('Pressure (ksi)')
plt.ylabel('S0')
plt.subplot(212)
plt.scatter(p_train, s0_train)
plt.scatter(p_test, s0_test)
plt.xlabel('Pressure (ksi)')
plt.ylabel('S0')

rs_m, rs_b = np.polyfit(p_test,rs_test, deg = 1)
y = rs_m * p_test + rs_b
plt.figure()
plt.subplot(211)
plt.scatter(o_train, rs_train)
plt.scatter(o_test, rs_test)

plt.xlabel('Orifice Diameter (in)')
plt.ylabel('RS')
plt.subplot(212)
plt.scatter(p_train, rs_train)
plt.scatter(p_test, rs_test)
plt.scatter(p_test_rs, rs_test_rs)
#plt.scatter(p_test_rs, rs_test_rs)
plt.plot(p_test, y)
plt.xlabel('Pressure (ksi)')
plt.ylabel('RS')






# %%
