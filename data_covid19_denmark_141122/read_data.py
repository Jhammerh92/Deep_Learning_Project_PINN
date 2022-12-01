
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO - Is the file below the right one to choose? We need to be able to answer these questions:
# TODO - How to find recovered?
# TODO - How to find infected? (we only get the day they become infected - no sense of time)
# TODO - How to think about deaths (probably cumulative, but how vs infected?)

lockdown1 = pd.Timestamp(year=2020,month=3,day=11)
lockdown2 = pd.Timestamp(year=2021,month=12,day=21)
fake_lockdown = pd.Timestamp(year=2022,month=2,day=21)

region_hovedstaden_indbyggere = 1867948

file = '03_bekraeftede_tilfaelde_doede_indlagte_pr_dag_pr_koen.csv'
df = pd.read_csv(r'C:\Users\willi\Documents\GitHub\Deep_Learning_Project_PINN\data_covid19_denmark_141122\Regionalt_DB\\'+file
                 , sep  =";", encoding= 'iso-8859-1')
df = df.drop('Køn', axis=1)
df['Prøvetagningsdato'] = pd.to_datetime(df['Prøvetagningsdato'])

df = df.groupby(['Region', 'Prøvetagningsdato']).sum()

df = df.loc['Hovedstaden']
df = df[df.index > lockdown2]


# plt.figure()
# plt.plot(np.arange(len(df.index)), df['Bekræftede tilfælde i alt'], label='Cases')
# plt.grid()

#plt.figure()
#plt.plot(np.arange(len(df.index)), df['Døde'], label='Death')

df['DødeCumulative'] = df['Døde'].cumsum()
df['susceptible'] =region_hovedstaden_indbyggere - df['Kummuleret antal bekræftede tilfælde'] - df['Kummuleret antal døde']

df = df[df.index < fake_lockdown]

data = {}
for i, (date, row) in enumerate(df.iterrows()):
    data[date] = df.iloc[i-14:i]['Bekræftede tilfælde i alt'].sum()

df['infected'] = pd.Series(data)

plt.figure()
plt.plot(df.index,df['susceptible'])
plt.plot(df.index,df['DødeCumulative'])
plt.plot(df.index,df['infected'])
plt.show()


days = 50
prob = 0.3
num_infected = 10000
infected = np.ones((num_infected,days))
recovered = 0
for person in range(num_infected):
    for day in range(days):
        if np.random.random(1)[0] < prob:
            infected[person][day:] = 0
            break    
plt.plot(infected.mean(axis=0))

