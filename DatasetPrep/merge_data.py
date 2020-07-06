import pandas as pd
import numpy as np

master_file = 'BhajanLabels.xlsx'
incoming_file = 'BhajanLabels-Sneha.xlsx'
updated_master_file = 'BhajanLabels-Updated.xlsx'

df_master = pd.read_excel(master_file)
df_incoming = pd.read_excel(incoming_file)

df_incoming.set_index('UUID',inplace=True)
df_incoming = df_incoming[['Lead Segments','Chorus Segments']].copy()
segments_dict = df_incoming.to_dict('index')

for k in segments_dict:
    df_master.loc[df_master.UUID == k,['Lead Segments','Chorus Segments']] = [segments_dict[k]['Lead Segments'],segments_dict[k]['Chorus Segments']]

df_master.to_excel(updated_master_file,index=False)