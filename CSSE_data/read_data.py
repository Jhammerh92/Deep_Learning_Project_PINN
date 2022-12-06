
import os
import pandas as pd
import matplotlib.pyplot as plt

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

dk_population = 5.857*10**(6)
#dk_population = 3*10**(5)
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