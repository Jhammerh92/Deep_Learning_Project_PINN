
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from
# https://github.com/CSSEGISandData/COVID-19

def select_denmark(df):
    if 'Country/Region' in df.columns:
        df = df.rename({'Country/Region':'Country_Region'}, axis=1)
    if 'Province/State' in df.columns:
        df = df.rename({'Province/State':'Province_State'}, axis=1)
        
    df = df[(df['Country_Region'] == 'Denmark')]
    df = df[pd.isnull(df['Province_State'])]
    return df

def read_data():
    dfs = []
    for file in os.listdir('daily'):
        df = pd.read_csv('daily\\'+file)
        if 'Active' not in df.columns:
            continue
        df = select_denmark(df)
        df = df[['Confirmed','Deaths','Recovered','Active']] #,'Combined_Key','Case_Fatality_Ratio']]
        date = file[:-4]
        df.index = [pd.to_datetime(date, format='%m-%d-%Y')]
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs

raw_data = read_data()
#%%
data = raw_data.copy()

# dk_population = 5.857*10**(6)
dk_population = 3*10**(5)
data['S'] = dk_population - data['Active'] - data['Recovered'] - data['Deaths']
data = data.dropna(subset='S')

#%%
fig, ax = plt.subplots(dpi=200)
s = 3
plt.scatter(data['S'].index, data['S'], label='S', color='C0',s=s)
# plt.scatter(data['Confirmed'].index, data['Confirmed'], label='C',color='C5',s=s)
plt.scatter(data['Active'].index, data['Active'], label='I', color='C1',s=s)
plt.scatter(data['Recovered'].index, data['Recovered'], label='R', color='C2',s=s)
plt.scatter(data['Deaths'].index, data['Deaths'], label='D', color='C3',s=s)
# plt.xlim(min(data.index), pd.Timestamp(year=2021,month=4,day=1))
# plt.xlim(pd.Timestamp(year=2020,month=8,day=1), pd.Timestamp(year=2021,month=4,day=1))
plt.xlim(pd.Timestamp(year=2020,month=8,day=1), pd.Timestamp(year=2021,month=4,day=1))
fig.autofmt_xdate()
plt.grid()
plt.legend()

#%%

df_subset = data[data.index > pd.Timestamp(year=2020,month=8,day=1)]
df_subset = df_subset[df_subset.index < pd.Timestamp(year=2021,month=4,day=1)]
df_subset = df_subset.loc[df_subset.index.sort_values()]

wsol = df_subset[['S','Active','Recovered','Deaths']].values
t = np.arange(len(df_subset.index))

#%%
t_bool = t < max(t)
wsol_subset = wsol[t_bool]
t_subset = wsol[t_bool]

#%%
import sys
sys.path.insert(0, r'C:\Users\willi\Documents\GitHub\Deep_Learning_Project_PINN\DINN_implementation')

from SIRD_deepxde_class import SIRD_deepxde_net
model = SIRD_deepxde_net(t, wsol, init_num_people=dk_population, model_name='dk_data')
model.init_model(print_every=100)

#%%
model.train_model(iterations=60000, print_every=100)

#%%
import ODE_SIR
solver = ODE_SIR.ODESolver()

alpha_nn, beta_nn, gamma_nn = model.get_best_params()
t_nn_param, wsol_nn_param, N_nn_param = solver.solve_SIRD(alpha_nn, beta_nn, gamma_nn,
                                                          init_num_people=dk_population+wsol[0,1],
                                                          I=wsol[0,1],R=wsol[0,2],D=wsol[0,3],
                                                          numpoints=241)
#%%
fig, ax = plt.subplots(dpi=300, figsize=(8,3))

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax.plot(wsol[:,0],color='C0', label='Train data')
ax.plot(wsol[:,1],color='C1')
ax.plot(wsol[:,2],color='C2')
ax.plot(wsol[:,3],color='C3')

ax.plot(wsol_nn_param[:,0], linestyle='--',color='C0', label='Prediction')
ax.plot(wsol_nn_param[:,1], linestyle='--',color='C1')
ax.plot(wsol_nn_param[:,2], linestyle='--',color='C2')
ax.plot(wsol_nn_param[:,3], linestyle='--',color='C3')

ax.grid(linestyle=':') #
ax.set_axisbelow(True)
ax.set_xlabel('Time [day]')
ax.set_ylabel('Number of people')
ax.legend()
ax.set_title('Denmark Covid-19 data')
plt.tight_layout()
plt.savefig('DK data',bbox_inches='tight')

#%%
import deepxde as dde
dde.utils.external.plot_loss_history(model.losshistory,fname='dk_data_loss')
