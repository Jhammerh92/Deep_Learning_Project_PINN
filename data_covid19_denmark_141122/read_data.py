
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO - Is the file below the right one to choose? We need to be able to answer these questions:
# TODO - How to find recovered?
# TODO - How to find infected? (we only get the day they become infected - no sense of time)
# TODO - How to think about deaths (probably cumulative, but how vs infected?)

lockdown1 = pd.Timestamp(year=2020,month=3,day=11)
lockdown2 = pd.Timestamp(year=2021,month=12,day=21)


file = '03_bekraeftede_tilfaelde_doede_indlagte_pr_dag_pr_koen.csv'
df = pd.read_csv(r'C:\Users\willi\Documents\GitHub\Deep_Learning_Project_PINN\data_covid19_denmark_141122\Regionalt_DB\\'+file
                 , sep  =";", encoding= 'iso-8859-1')
df = df.drop('Køn', axis=1)

df = df.groupby(['Region', 'Prøvetagningsdato']).sum()

df = df.loc['Hovedstaden']

plt.figure()
plt.plot(np.arange(len(df.index)), df['Bekræftede tilfælde i alt'], label='Cases')
plt.grid()

#plt.figure()
#plt.plot(np.arange(len(df.index)), df['Døde'], label='Death')


data = {}
for i, (date, row) in enumerate(df.iterrows()):
    data[date] = df.iloc[i-7:i]['Bekræftede tilfælde i alt'].sum()

# plt.figure()
plt.plot(np.arange(len(df.index)),pd.Series(data))
plt.show()