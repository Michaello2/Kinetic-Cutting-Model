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

    v_sep = mn*l*mt*ph*s0*r/rs*(psi_max/(1+r/(rs*r0)))**2
    return v_sep

def s0rs_opt(args, para, v_sep):
    v_sepmod = kin_equ(para, args)
    error = np.sum((v_sepmod/v_sep - 1)**2)
    return error

def s0rs(vals, v_sep):
    mat, l, mt, o, p, afr, psi_max, r0 = vals;
    guess = np.array([1,1])
    inp = ([mat, l, mt, o, p, afr, psi_max, r0])
    optim_all = sp.optimize.fmin(func = s0rs_opt, x0 = guess, args = (inp, v_sep,))
    return optim_all

def ind_reg(data):
    filt_size = data.shape
    save = []
    index = 0;
    sort_index = 0

    for i in range (0,filt_size[0],1):
        data['Material'].iloc[i] = machine_num(data['Material'].iloc[i])
        if data['Material'].iloc[index] == data['Material'].iloc[i] and math.isclose(data['Thickness'].iloc[index], data['Thickness'].iloc[i], abs_tol= .02) == True and data['Mixing Tube Diameter'].iloc[index] == data['Mixing Tube Diameter'].iloc[i] and data['Orifice'].iloc[index] == data['Orifice'].iloc[i] and data['Pressure'].iloc[index] == data['Pressure'].iloc[i]:
            data['Sorting'].iloc[i] = sort_index;
            
        else:
            index = i;
            sort_index = sort_index + 1
            data['Sorting'].iloc[i] = sort_index;
    
    for i in range(0,sort_index,1):
        section  = data.loc[data["Sorting"] == i]
        data_size = section.shape
        if data_size[0] > 0:
            #Material properties and cutting parameters.
            mat = section['Material'].iloc[0];
            l = section['Thickness'];
            mt = section['Mixing Tube Diameter'];
            o = section['Orifice'];
            p = section['Pressure'];
            afr = section["Actual Abrasive"];
            psi_max = .68;
            r0 = .57;
            para = [mat, l, mt, o, p, afr, psi_max, r0]

            #Modeling values
            v_sep = section['Experimental Separation']
            s0, rs = s0rs(para, v_sep)

            args = [s0, rs]
            v_sepmod = kin_equ(para, args)

            save.append([s0, rs, l.iloc[0], o.iloc[0], mt.iloc[0], o.iloc[0], p.iloc[0]])

    return save, 


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

#sort data into by material type, material thickness, mixing tube diameter, orifice diameter, pressure, and abrasive feed rate.
data_sep = train_data.sort_values(by = ['Material','Thickness','Mixing Tube Diameter','Orifice', 'Pressure', 'Actual Abrasive'], ascending = [True, True, True, True, True, True])
data_filt = data_sep[data_sep['Mixing Tube Diameter'] == .03]
#data_filt = data_sep[data_sep['Pressure'] == 80]

filt_size = data_filt.shape
index = 0;
sort_index = 0
for i in range (0,filt_size[0],1):
    data_filt['Material'].iloc[i] = machine_num(data_filt['Material'].iloc[i])   

    if data_filt['Material'].iloc[index] == data_filt['Material'].iloc[i] and math.isclose(data_filt['Thickness'].iloc[index], data_filt['Thickness'].iloc[i], abs_tol= .02) == True and data_filt['Mixing Tube Diameter'].iloc[index] == data_filt['Mixing Tube Diameter'].iloc[i] and data_filt['Orifice'].iloc[index] == data_filt['Orifice'].iloc[i] and data_filt['Pressure'].iloc[index] == data_filt['Pressure'].iloc[i]:
        data_filt['Sorting'].iloc[i] = sort_index;
        
    else:
        index = i;
        sort_index = sort_index + 1
        data_filt['Sorting'].iloc[i] = sort_index;
save = []

ax0 = plt.figure().add_subplot(projection='3d')

plt.figure()
plt.title('training data set')

for i in range(0,sort_index,1):
    data  = data_filt.loc[data_filt["Sorting"] == i]
    data_size = data.shape
    if data_size[0] > 0:
        #Material properties and cutting parameters.
        mat = data['Material'].iloc[0];
        l = data['Thickness'];
        mt = data['Mixing Tube Diameter'];
        o = data['Orifice'];
        p = data['Pressure'];
        afr = data["Actual Abrasive"];
        psi_max = .68;
        r0 = .57;
        para = [mat, l, mt, o, p, afr, psi_max, r0]

        #Modeling values
        v_sep = data['Experimental Separation']
        s0, rs = s0rs(para, v_sep)

        args = [s0, rs]
        v_sepmod = kin_equ(para, args)

        save.append([s0, rs, l.iloc[0], mat, mt.iloc[0], o.iloc[0], p.iloc[0]])

        plt.scatter(afr,v_sep)
        plt.plot(afr,v_sepmod)


        if o.iloc[0] == .01:
            col= 'green'
        elif o.iloc[0] == .011:
            col = 'cyan'
        elif o.iloc[0] == .012:
            col = 'magenta'
        elif o.iloc[0] == .013:
            col = 'yellow'
        elif o.iloc[0] == .014:
            col = 'red'
        elif o.iloc[0] == .015:
            col = 'blue'
        elif o.iloc[0] == .016:
            col = 'black'
        ax0.scatter(o.iloc[0],p.iloc[0],s0, color = col)

plt.xlabel('abrasive feed rate (lb/min)')
plt.ylabel('separation speed (in)') 
save = np.array(save, dtype = float)
s0, rs, l, o, mt, o, p = [save[:,0], save[:,1],save[:,2],save[:,3],save[:,4],save[:,5],save[:,6]]

#plot data
fig1, ax1 = plt.subplots(3,1)
ax1[0].set_title('Scaling Constant S0')
fig1.tight_layout()
ax1[0].scatter(o, s0)
ax1[0].set_xlabel('Orifice Diameter (in)')
ax1[1].scatter(p, s0)
ax1[1].set_xlabel('Pressure (ksi)')
ax1[2].scatter(hyd_pow(o,p),s0)
ax1[2].set_ylabel('Scaling Constant S')
ax1[2].set_xlabel('hydarulic power (hp)')

fig2, ax2 = plt.subplots(3,1)
ax2[0].set_title('Shifting Constant R_s')
fig2.tight_layout()
ax2[0].scatter(o, rs)
ax2[0].set_xlabel('Orifice Diameter (in)')
ax2[1].scatter(p, rs)
ax2[1].set_xlabel('Pressure (ksi)')
ax2[2].scatter(hyd_pow(o,p), rs)
ax2[2].set_ylabel('Shifting Constant r_s')
ax2[2].set_xlabel('hydarulic power (hp)')



#machine learning model
model = LinearRegression()
model1 = KNeighborsRegressor()
model2 = DecisionTreeRegressor()

l_in = 1;
mat_in = 'Aluminum 6061'
o_in = .01;
mt_in = .030;
p_in = 80;
psi_max = .68;
r0 = .57;

#Test Data:
data_test = data_filt[data_sep['Thickness'] == l_in]
data_test = data_test[data_test['Material'] == machine_num(mat_in)]
data_test = data_test[data_test['Orifice'] == o_in]
data_test = data_test[data_test["Mixing Tube Diameter"] == mt_in]
data_test = data_test[data_test['Pressure'] == p_in]

afr = data_test['Actual Abrasive']

mod_x = save[:,2:]
mod_y = save[:,:2]

model.fit(mod_x, mod_y)
model1.fit(mod_x, mod_y)
model2.fit(mod_x, mod_y)

mod_input = [l_in, machine_num(mat_in), mt_in, o_in, p_in]

prediction = model.predict([mod_input])
prediction1 = model1.predict([mod_input])
prediction2 = model2.predict([mod_input])

s0_pred = float(prediction[0,0])
rs_pred = float(prediction[0,1])

s0_pred1 = float(prediction1[0,0])
rs_pred1 = float(prediction1[0,1])

s0_pred2 = float(prediction2[0,0])
rs_pred2 = float(prediction2[0,1])

para = [machine_num(mat_in), l_in, mt_in, o_in, p_in, afr, psi_max, r0]
args = [s0_pred, rs_pred]
v_sepmod = kin_equ(para, args)

args = [s0_pred1, rs_pred1]
v_sepmod1 = kin_equ(para, args)

args = [s0_pred2, rs_pred2]
v_sepmod2 = kin_equ(para, args)

plt.figure()
plt.plot(afr,v_sepmod, label = 'lin reg')
plt.plot(afr,v_sepmod1, label = 'KN reg')
plt.plot(afr,v_sepmod2, label = 'tree reg')
plt.scatter(afr,data_test['Experimental Separation'])
plt.xlabel('Abrasive Feed Rate (lb/min)')
plt.ylabel('Separation Speed (ipm)')
plt.legend()

error = [];
error1 = [];
error2 = [];

plt.figure()
for i in range(0,filt_size[0],1):

    mod_input = [data_filt['Thickness'].iloc[i],
                 data_filt['Material'].iloc[i],
                 data_filt['Mixing Tube Diameter'].iloc[i],
                 data_filt['Orifice'].iloc[i],
                 data_filt['Pressure'].iloc[i]]
    
    prediction = model.predict([mod_input])
    prediction1 = model1.predict([mod_input])
    prediction2 = model2.predict([mod_input])

    s0_pred = float(prediction[0,0])
    rs_pred = float(prediction[0,1])

    s0_pred1 = float(prediction1[0,0])
    rs_pred1 = float(prediction1[0,1])

    s0_pred2 = float(prediction2[0,0])
    rs_pred2 = float(prediction2[0,1])

    para = [data_filt['Material'].iloc[i],
            data_filt['Thickness'].iloc[i],
            data_filt['Mixing Tube Diameter'].iloc[i],
            data_filt['Orifice'].iloc[i],
            data_filt['Pressure'].iloc[i],
            data_filt['Actual Abrasive'].iloc[i],
            psi_max,
            r0]
            
    args = [s0_pred, rs_pred]
    v_sepmod = kin_equ(para, args)

    args = [s0_pred1, rs_pred1]
    v_sepmod1 = kin_equ(para, args)

    args = [s0_pred2, rs_pred2]
    v_sepmod2 = kin_equ(para, args)

    error.append(v_sepmod/data_sep['Experimental Separation'].iloc[i]-1);
    error1.append(v_sepmod1/data_sep['Experimental Separation'].iloc[i]-1);
    error2.append(v_sepmod2/data_sep['Experimental Separation'].iloc[i]-1);

    plt.scatter(hyd_pow(data_filt['Orifice'].iloc[i],data_filt['Pressure'].iloc[i]),s0_pred, color = 'blue')
    plt.scatter(hyd_pow(data_filt['Orifice'].iloc[i],data_filt['Pressure'].iloc[i]),s0_pred1, color = 'green')
    plt.scatter(hyd_pow(data_filt['Orifice'].iloc[i],data_filt['Pressure'].iloc[i]),s0_pred2, color = 'orange')




plt.figure()
plt.title('Machine Learning Regression Effects on Error')
plt.scatter(data_filt['Actual Abrasive'], error, label = 'lin reg')
plt.scatter(data_filt['Actual Abrasive'], error1, label = 'KN reg')
plt.scatter(data_filt['Actual Abrasive'], error2, label = 'tree reg')
plt.xlabel('Abrasive Feed Rate (lb/min)')
plt.ylabel('Error')
plt.legend()

l_test = 1;
mat_test = 'Aluminum 6061'
o_test = np.linspace(.01,.020,20)
mt_test = .03
p_test = np.linspace(40,85,20)

ax3 = plt.figure().add_subplot(projection='3d')
for i in range(0,20,1):
    for j in range(0,20,1):
        mod_input = [l_test, machine_num(mat_test), mt_test, o_test[i], p_test[j]]
        prediction = model.predict([mod_input])
        prediction1 = model1.predict([mod_input])
        prediction2 = model2.predict([mod_input])

        s0_pred = float(prediction[0,0])
        rs_pred = float(prediction[0,1])

        s0_pred1 = float(prediction1[0,0])
        rs_pred1 = float(prediction1[0,1])

        s0_pred2 = float(prediction2[0,0])
        rs_pred2 = float(prediction2[0,1])

        if math.isclose(o_test[i],.01):
            col = 'blue'
        elif math.isclose(o_test[i], .011):
            col = 'orange'
        elif math.isclose(o_test[i], .012):
            col = 'cyan'
        elif math.isclose(o_test[i], .013):
            col = 'green'
        elif math.isclose(o_test[i], .014):
            col = 'red'
        elif math.isclose(o_test[i], .015):
            col = 'black'

        ax3.scatter(o_test[i],p_test[j],s0_pred, color = 'green')
        #ax3.scatter(o_test[i],p_test[j],s0_pred1, color = 'blue')
        #ax3.scatter(o_test[i],p_test[j],s0_pred2, color = 'cyan')

ax3.scatter(o,p,s0,color = 'black')
ax3.set_xlabel('Orifice Diameter (in)')
ax3.set_ylabel('Pressure (ksi)')



print(prediction)





# %%
