
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO - Is the file below the right one to choose? We need to be able to answer these questions:
# TODO - How to find recovered?
# TODO - How to find infected? (we only get the day they become infected - no sense of time)
# TODO - How to think about deaths (probably cumulative, but how vs infected?)

file = '03_bekraeftede_tilfaelde_doede_indlagte_pr_dag_pr_koen.csv'
df = pd.read_csv(r'C:\Users\willi\Documents\GitHub\Deep_Learning_Project_PINN\data_covid19_denmark_141122\Regionalt_DB\\'+file
                 , sep  =";", encoding= 'iso-8859-1')
df = df.drop('Køn', axis=1)

df = df.groupby(['Region', 'Prøvetagningsdato']).sum()

df = df.loc['Hovedstaden']

plt.plot(np.arange(len(df.index)), df['Bekræftede tilfælde i alt'], label='Cases')

plt.plot(np.arange(len(df.index)), df['Døde'], label='Death')

