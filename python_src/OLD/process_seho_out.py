from branca import colormap
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import branca
from shapely.geometry import Polygon
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt

from os import walk
from os.path import join
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
                      'LatRx', 'LonRx', 'AltRx',                    # LLS Reciever
                      'TxID',  'LatTx', 'LonTx', 'AltTx',           # Transmitter ID, LLS Transmitter
                      'LatSp', 'LonSp', 'AltSp',                    # LLS Specular point
                      'LosDAz', 'LosDEv', 'LosRAz', 'LosREv',       # Line of sight Direct signal (Azimuth, Elevation); Line of sight Reflected signal (Azimuth, Elevation)
                      'Incidence', 'Elevation',                     # Incidence angle, Elevation angle
                      'LandMask', 'idk']                            # Landmask (1:land, 0:water), extra unknown column
    unused_columns = ['LatRx', 'LonRx', 'AltRx',
                      'LatTx', 'LonTx', 'AltTx', 
                      'LosDAz', 'LosDEv', 'LosRAz', 'LosREv',
                      'Incidence', 'Elevation',
                      'idk']
    list_dfs = []

    for file_name in file_names:
        try:
            data = pd.read_csv(file_name, sep=" ", header=None)
        except pd.errors.EmptyDataError:
            continue
        data.columns = column_names
        data = data.drop(columns=unused_columns)

        list_dfs = list_dfs + [data]

    df = pd.concat(list_dfs)

    # Remove water samples (land=1, water=0)
    df = df[df.LandMask == 1]

    return df

def get_land_latlon(land_mask_dir):
    '''
    From saved latlon h5 file, get latitude and longitude grid with 1km spacing that 
    '''
    file_name = land_mask_dir + 'land_latlon.h5'
    latlon = []

    with h5py.File(file_name, 'r') as f:
        group_key = list(f.keys())[0]
        print(group_key)
        print(list(f[group_key]))
        print(list(f[group_key]['axis1']))

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

def get_revisit_info(all_specular_df, transmitters):
    '''
        Returns array with the revisit info

        Inputs:
            specular_df (Pandas DF): Dataframe which contains lat, lon specular points
            transmitters (List of tuples): Contains which transmitters we are interested. If empty, consider all transmitters
    '''
    # Remove transmitters that we don't want to consider
    specular_df = pd.DataFrame()
    if transmitters:
        print('Trimming off extra transmitters...')
        for trans_set in transmitters:
            # Lower bound
            specular_df = specular_df.append(all_specular_df[all_specular_df['TxID'] >= trans_set[0]])
            # Upper bound
            specular_df = specular_df[specular_df['TxID'] <= trans_set[1]]
    else:
        # If transmitters is empty, we consider all transmitters
        specular_df = all_specular_df

    # Possible that a transmitter constellation is never used...
    if specular_df.empty:
        exit('This set of transmitters is never used to generate a specular point. Please select another set.')
    
    # Round lat and long and then use groupby to throw them all in similar buckets
    specular_df['approx_LatSp'] = round(specular_df['LatSp'],1)
    specular_df['approx_LonSp'] = round(specular_df['LonSp'],1)

    # Calculate time difference
    specular_df.sort_values(by=['approx_LatSp', 'approx_LonSp', 'JulianDay'], inplace=True)
    specular_df['revisit'] = specular_df['JulianDay'].diff()

    # Correct for borders
    specular_df['revisit'].mask(specular_df.approx_LatSp != specular_df.approx_LatSp.shift(1), other=np.nan, inplace=True)
    specular_df['revisit'].mask(specular_df.approx_LonSp != specular_df.approx_LonSp.shift(1), other=np.nan, inplace=True)

    # Get max revisit and store in new DF
    indeces = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])['revisit'].transform(max) == specular_df['revisit']

    max_rev_area_df = specular_df[indeces]

    # Get rid of extra columns
    # Any revisit that is less than 1 hour is removed. Typically this occurs because of a lack of samples (due to low sim time)
    extra_cols = ['JulianDay', 'LatSp', 'LonSp', 'AltSp', 'LandMask']
    max_rev_area_df['revisit'].mask(max_rev_area_df['revisit'] < 0.04, other=np.nan, inplace=True)
    max_rev_area_df.drop(extra_cols, inplace=True, axis=1)

    return max_rev_area_df

def plot_revisit_heatmap(max_rev_area_df):
    # Remove NaNs
    max_rev_area_df = max_rev_area_df[max_rev_area_df['revisit'].notnull()]

    # Heat map
    hmap = folium.Map(location=[42.5, -80], zoom_start=7, )
    hm_wide = HeatMap(list(zip(max_rev_area_df.approx_LatSp.values, max_rev_area_df.approx_LonSp.values, max_rev_area_df.revisit.values)),
                      gradient={0.0: '#00ae53', 0.2: '#86dc76', 0.4: '#daf8aa',
                                0.6: '#ffe6a4', 0.8: '#ff9a61', 1.0: '#ee0028'})
    hmap.add_child(hm_wide)

    max_amt = max(max_rev_area_df.revisit.values)

    print(max_amt)

    colormap = branca.colormap.StepColormap(
               colors=['#00ae53', '#86dc76', '#daf8aa',
                       '#ffe6a4', '#ff9a61', '#ee0028'],
               vmin=0,
               vmax=max_amt,
               index=[0, 2, 4, 6, 8, 10])
    colormap.caption='Revisit Time'
    colormap.add_to(hmap)

    hmap.save('test_revisit_10day.html')

def plot_revisit_map_2(max_rev_area_df, map_name='test'):
    # First reduce the resolution of the polymap to avoid murdering my computer
    # Round lat and long and then use groupby to throw them all in similar buckets
    max_rev_area_df['approx_LatSp'] = round(max_rev_area_df['approx_LatSp'])
    max_rev_area_df['approx_LonSp'] = round(max_rev_area_df['approx_LonSp'])

    # Get max revisit and store in new DF
    indeces = max_rev_area_df.groupby(['approx_LatSp', 'approx_LonSp'])['revisit'].transform(max) == max_rev_area_df['revisit']

    max_rev_area_df = max_rev_area_df[indeces]

    # Now generate the map
    map = folium.Map(location=[42.5, -80], zoom_start=7, )

    # Remove NaNs
    max_rev_area_df = max_rev_area_df[max_rev_area_df['revisit'].notnull()]
    # Generate Polygons
    max_rev_area_df['geometry'] = max_rev_area_df.apply(lambda row: Polygon([(row.approx_LonSp-0.5, row.approx_LatSp-0.5), 
                                                                             (row.approx_LonSp+0.5, row.approx_LatSp-0.5),
                                                                             (row.approx_LonSp+0.5, row.approx_LatSp+0.5),
                                                                             (row.approx_LonSp-0.5, row.approx_LatSp+0.5)]), axis=1)
    max_amt = max(max_rev_area_df.revisit.values)
    print('Max revisit value: ', max_amt)
    # colormap_dept = branca.colormap.StepColormap(
    #     colors=['#0A2F51', '#0E4D64', '#137177', '#188977', '#1D9A6C',
    #             '#39A96B', '#56B870', '#74C67A', '#99D492', '#BFE1B0', '#DEEDCF'],
    #     vmin=0,
    #     vmax=max_amt,
    #     index=[0,1,2,3,4,5,6,7,8,9,10])
    colormap_dept = branca.colormap.LinearColormap(colors=['green','yellow', 'red'], vmin=0, vmax=max_amt)

    for _, r in tqdm(max_rev_area_df.iterrows(), total=max_rev_area_df.shape[0]):
        # print(r['revisit'])
        style_func = lambda x, revisit=r['revisit']: {
            'fillColor': colormap_dept(revisit),
            'color': '',
            'weight': 1.0,
            'fillOpacity': 0.5
        }
        # print(style_func(r['revisit']))
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j, style_function=style_func, overlay=True, control=True)
        geo_j.add_to(map)
        # break
    
    # Add legend
    colormap_dept.caption='Revisit Time'
    colormap_dept.add_to(map)
    # map.add_child(folium.LayerControl())

    # Save it
    map.save(map_name+'.html')

    # return max_rev_area_df

def plot_revisit_stats(max_rev_area_df, plot_title='Frequency Distribution of Maximum Revisit Time'):
    '''
        Get relevant revisit statistics
    '''
    # Remove NaNs
    max_rev_area_df = max_rev_area_df[max_rev_area_df['revisit'].notnull()]

    # Plot over all areas
    print('Creating histogram')
    print(max_rev_area_df)
    ax = max_rev_area_df['revisit'].plot.hist(bins=50, alpha=0.5)
    ax.plot()
    plt.xlabel('Maximum Revisit Time (days)')
    plt.title(plot_title)
    plt.show()


if __name__ == "__main__":
    # This path assumes all files in the folder are for this test. It does remove .gitignore files though
    path_to_output='/home/polfr/Documents/dummy_data/09_24_2021_10day/Unzipped/'
    file_names = get_all_files_dir(path_to_output)
    specular_df = get_files_pd(file_names)

    transmitters = [(49,78)]
    max_rev_area_df = get_revisit_info(specular_df, transmitters)
    plot_revisit_map_2(max_rev_area_df, map_name='polymap_hw05_GPS_10day')
    plot_revisit_stats(max_rev_area_df, plot_title='Frequency Distribution of Maximum Revisit Time\n GPS - 10 days - 0.1$^\circ$x0.1$^\circ$')

    # get_land_latlon('/home/polfr/Documents/dummy_data/data/')