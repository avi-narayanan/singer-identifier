import pandas as pd
import os
from shutil import copyfile


raw_data = os.path.join('.','RawData')
data_folder = os.path.join('.','Data')
master_file = 'BhajanLabels.xlsx'
df_master = pd.read_excel(master_file)

if not os.path.exists(data_folder):
    os.mkdir(data_folder)

df_master.set_index('File Name',inplace=True)
artists = df_master[(df_master.Artist != 'Other') & (df_master.Comment.isnull())].Artist
l = artists.size
for i,k in enumerate(artists.index):
    print("Copying",i+1,"of",l,end='\r')
    dest_folder = os.path.join(data_folder,artists[k])
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    src = os.path.join(raw_data,k)
    dst = os.path.join(dest_folder,k)
    copyfile(src, dst)
print("\nDone")