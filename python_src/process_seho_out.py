from branca import colormap
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import branca
from shapely.geometry import Polygon

from os import listdir, walk
from os.path import isfile,join
from tqdm import tqdm

from typing import List

## File which is used to process Seho's output


def get_all_files_dir(path_to_dir: str) -> List[str]:
    '''
        Gets all the file names in a directory

        Inputs:
            path_to_dir (str): full path to directory
    '''
    # file_names = [join(path_to_dir,f) for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]
    
    file_names = []
    for (dirpath, dirnames, filenames) in walk(path_to_dir):
        temp = [join(dirpath, f) for f in filenames]
        file_names.extend(temp)

    # Remove .gitignore files
    for name in file_names:
        if 'gitignore' in name:
            file_names.remove(name)

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
                      'LandMask', 'idk']
    unused_columns = ['LatRx', 'LonRx', 'AltRx',
                      'TxID', 'LatTx', 'LonTx', 'AltTx', 'idk']
    list_dfs = []

    for file_name in file_names:
        data = pd.read_csv(file_name, sep=" ", header=None)
        data.columns = column_names
        data = data.drop(columns=unused_columns)

        list_dfs = list_dfs + [data]

    df = pd.concat(list_dfs)

    # Remove water samples (land=1, water=0)
    df = df[df.LandMask == 1]

    return df

def get_specular_heatmap(specular_df):
    ''''
        Heatmap of the specular points
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

    # Generate Polygons
    df['geometry'] = df.apply(lambda row: Polygon([(row.longitude-0.05, row.latitude-0.05), 
                                                   (row.longitude+0.05, row.latitude-0.05),
                                                   (row.longitude+0.05, row.latitude+0.05),
                                                   (row.longitude-0.05, row.latitude+0.05)]), axis=1)
    print(df.head())
    # Heat map
    hmap = folium.Map(location=[42.5, -80], zoom_start=7, )
    hm_wide = HeatMap(list(zip(df.latitude.values, df.longitude.values, df.countSp.values)),
                      gradient={0.0: '#00ae53', 0.2: '#86dc76', 0.4: '#daf8aa',
                                0.6: '#ffe6a4', 0.8: '#ff9a61', 1.0: '#ee0028'},
                                )
    hmap.add_child(hm_wide)

    colormap = branca.colormap.StepColormap(
               colors=['#00ae53', '#86dc76', '#daf8aa',
                       '#ffe6a4', '#ff9a61', '#ee0028'],
               vmin=0,
               vmax=25,
               index=[0, 4, 8, 12, 16, 20])
    # colormap = colormap.to_step(index=[0,2, 4, 6, 8, 10, 12])
    colormap.caption='Number of Specular Points'
    colormap.add_to(hmap)
    # colormap_dept = branca.colormap.StepColormap(
    #     colors=['#00ae53', '#86dc76', '#daf8aa',
    #         '#ffe6a4', '#ff9a61', '#ee0028'],
    #     vmin=0,
    #     vmax=max_amt,
    #     index=[0, 2, 4, 6, 8, 10, 12])
    
    # style_func = lambda x: {
    #     'fillColor': colormap_dept(x['countSp']),
    #     'color': '',
    #     'weight': 0.0001,
    #     'fillOpacity': 0.1
    # }

    # folium.GeoJson(
    #     df,
    #     style_function=style_func,
    # ).add_to(hmap)

    hmap.save('test.html')

def get_revisit_info(specular_df, sim_start, sim_end):
    '''
        Returns array with the revisit info
    '''
    # Group the measurements into buckets
    # Round lat and long and then use groupby to throw them all in similar buckets
    specular_df['approx_LatSp'] = round(specular_df['LatSp'],1)
    specular_df['approx_LonSp'] = round(specular_df['LonSp'],1)

    grouped = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])

    buckets = pd.DataFrame(columns=['bucket_LatSp', 'bucket_LonSp', 'SpPoints', 'Revisits', 'MaxRevisit'])

    for key in tqdm(list(grouped.groups.keys())):
        group = grouped.get_group(key)
        times = group['JulianDay'].tolist()
        times = sorted(times + [sim_end, sim_start])
        diff = [t - s for s, t in zip(times, times[1:])]
        temp = {'bucket_LatSp': key[0], 'bucket_LonSp': key[1], 'SpPoints': group.to_dict(), 'Revisits': diff, 'MaxRevisit': max(diff)}
        buckets = buckets.append(temp, ignore_index=True)
    
    return buckets

def plot_revisit_heatmap(buckets):
    # Heat map
    hmap = folium.Map(location=[42.5, -80], zoom_start=7, )
    hm_wide = HeatMap(list(zip(buckets.bucket_LatSp.values, buckets.bucket_LonSp.values, buckets.MaxRevisit.values)),
                      gradient={0.0: '#00ae53', 0.2: '#86dc76', 0.4: '#daf8aa',
                                0.6: '#ffe6a4', 0.8: '#ff9a61', 1.0: '#ee0028'})
    hmap.add_child(hm_wide)

    print(max(buckets.MaxRevisit.values))

    # colormap = branca.colormap.StepColormap(
    #            colors=['#00ae53', '#86dc76', '#daf8aa',
    #                    '#ffe6a4', '#ff9a61', '#ee0028'],
    #            vmin=0,
    #            vmax=max_amt,
    #            index=[0, 2, 4, 6, 8, 10])
    # colormap = colormap.to_step(index=[0,2, 4, 6, 8, 10, 12])
    # colormap.caption='Number of Specular Points'
    # colormap.add_to(hmap)

    hmap.save('test_revisit.html')

if __name__ == "__main__":
    # This path assumes all files in the folder are for this test
    path_to='/home/polfr/Documents/dummy_data/09_17_2021/Unzipped/'
    file_names = get_all_files_dir(path_to)
    specular_df = get_files_pd(file_names)
    get_specular_heatmap(specular_df)
    # buckets = get_revisit_info(specular_df, sim_start=2459580.50000, sim_end=2459580.62500)
    # plot_revisit_heatmap(buckets)