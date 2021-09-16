import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import branca
import shapely.geometry

from os import listdir
from os.path import isfile,join

from typing import List

## File which is used to process Seho's output


def get_all_files_dir(path_to_dir: str) -> List[str]:
    '''
        Gets all the file names in a directory

        Inputs:
            path_to_dir (str): full path to directory
    '''
    file_names = [join(path_to_dir,f) for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]

    return file_names

def get_files_pd(file_names: List[str]) -> pd.DataFrame:
    '''
        Gets the files and returns a single Pandas Dataframe
        
        Inputs:
            file_names (List): list of all the full filenames
    '''
    column_names   = ['JulianDay', 
                      'LatRx', 'LonRx', 'AltRx',
                      'TxID', 'LatTx', 'LonTx', 'AltTx',
                      'LatSp', 'LonSp', 'AltSp',
                      'VSM', 'Sand', 'Clay', 'BulkDen', 'R_model', 'R_err', 'idk']
    unused_columns = ['LatRx', 'LonRx', 'AltRx',
                      'TxID', 'LatTx', 'LonTx', 'AltTx',
                      'VSM', 'Sand', 'Clay', 'BulkDen', 'R_model', 'R_err', 'idk']
    list_dfs = []

    for file_name in file_names:
        data = pd.read_csv(file_name, sep=" ", header=None)
        data.columns = column_names
        data = data.drop(columns=unused_columns)

        list_dfs = list_dfs + [data]

    df = pd.concat(list_dfs)

    return df

def data_test(df):
    df = df[df.LatSp > 40]
    df = df[df.LatSp < 41]
    df = df[df.LonSp > 30]
    df = df[df.LonSp < 31]

    print(df.info())

def get_good_heatmap(specular_df):
    ''''
        Trying to get revisit data from seho is a pain
    '''
    # Group the measurements into buckets
    # Round lat and long and then use groupby to throw them all in similar buckets
    specular_df['approx_LatSp'] = round(specular_df['LatSp'], 1)
    specular_df['approx_LonSp'] = round(specular_df['LonSp'], 1)

    test = specular_df.groupby(['approx_LatSp', 'approx_LonSp']).size()
    indeces = test.index.tolist()
    df = pd.DataFrame(indeces, columns=['latitude', 'longitude'])
    df['countSp'] = test.values.astype('float')

    max_amt = float(df.countSp.max())
    print(max_amt)

    # Heat map
    hmap = folium.Map(location=[42.5, -80], zoom_start=7, )
    hm_wide = HeatMap(list(zip(df.latitude.values, df.longitude.values, df.countSp.values)),
                      gradient={0.0: '#00ae53', 0.2: '#86dc76', 0.4: '#daf8aa',
                                0.6: '#ffe6a4', 0.8: '#ff9a61', 1.0: '#ee0028'})
    hmap.add_child(hm_wide)

    colormap = branca.colormap.StepColormap(
               colors=['#00ae53', '#86dc76', '#daf8aa',
                       '#ffe6a4', '#ff9a61', '#ee0028'],
               vmin=0,
               vmax=max_amt,
               index=[0, 2, 4, 6, 8, 10])
    # colormap = colormap.to_step(index=[0,2, 4, 6, 8, 10, 12])
    colormap.caption='Number of Specular Points'
    colormap.add_to(hmap)

    hmap.save('test.html')


if __name__ == "__main__":
    # This path assumes all files in the folder are for this test
    path_to='/home/polfr/Documents/aae450-moist/test_output/out/'
    file_names = get_all_files_dir(path_to)
    specular_df = get_files_pd(file_names)
    # data_test(specular_df)
    # generate_pass_heatmap(specular_df)

    get_good_heatmap(specular_df)