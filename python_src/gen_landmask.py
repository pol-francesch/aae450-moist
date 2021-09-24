import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

def get_landmask(land_mask_dir):
    land_mask_path = land_mask_dir+'Land_Mask_1km_EASE2_grid_150101_v004.h5'
    lat_path       = land_mask_dir+'EZ2Lat_M01_002_vec.float32'
    lon_path       = land_mask_dir+'EZ2Lon_M01_002_vec.float32'
    land_mask = []
    lat       = []
    lon       = []

    print('Getting land mask and latitude and longitude values...')
    with h5py.File(land_mask_path, 'r') as f:
        group_key = list(f.keys())[1]

        # Get the data
        land_mask = np.asarray(list(f[group_key]['mask']))

    lat = np.fromfile(lat_path, dtype=np.float32)
    lon = np.fromfile(lon_path, dtype=np.float32)

    latlon = np.array(np.meshgrid(lat,lon)).T.reshape(-1,2)

    return land_mask, latlon

def get_land_latlon(land_mask_dir):
    '''
    Don't run this. It currently breaks my computer
    '''
    land_mask, latlon  = get_landmask(land_mask_dir)
    latlon_mask = pd.DataFrame(data=latlon, columns=['lat', 'lon'])
    latlon_mask['land_mask'] = land_mask.reshape(-1,1).astype(np.uint8)

    print(latlon_mask.dtypes)

    n = 200000
    list_df = [latlon_mask[i:i+n] for i in range(0,latlon_mask.shape[0],n)]

    for df in tqdm(list_df):
        df = df[df['land_mask'] > 0]
        # print(df.head())
    
    latlon_mask = pd.concat(list_df)
    print(latlon_mask)