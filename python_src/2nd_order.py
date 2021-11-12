from operator import pos
import pandas as pd 
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from scipy.spatial import KDTree
from fqs.fqs import quartic_roots
from preprocess import interpolation, interpolation2

EARTH_RADIUS = 6371.009

def load_data(file_name, columns=None):
    data = np.loadtxt(file_name, skiprows=0, usecols=columns)

    return data

def get_spec(recx, recy, recz, transx, transy, transz):
    '''
        Given reciever and transmitter location, return specular point.
        Return empty array if no specular point is found.
        Reciever and transmitter locations are in the order of 1.

        Source: https://www.geometrictools.com/Documentation/SphereReflections.pdf
    '''
    global EARTH_RADIUS

    # Break down the inputs
    rec = np.array([recx, recy, recz]) / EARTH_RADIUS
    trans = np.array([transx, transy, transz]) / EARTH_RADIUS

    # Prework - dot products
    a = np.sum(rec*rec)
    b = np.sum(rec*trans)
    c = np.sum(trans*trans)

    # Step 1
    coeffs = [4*c*(a*c-b**2), -4*(a*c-b**2), a+2*b+c-4*a*c, 2*(a-b), a-1]
    roots = np.roots(coeffs)
    y = roots[roots > 0]
    
    if y.size == 0:
        return None
    
    # Step 2
    x = (-2*c*y**2 + y + 1) / (2*b*y + 1)

    if x[x > 0].size == 0:
        return None
    i = np.argwhere(x > 0)[0][0]

    spec = (x[i]*rec + y[i]*trans)*EARTH_RADIUS

    return spec[0], spec[1], spec[2]

def get_spec_vectorized(recx, recy, recz, transx, transy, transz, time):
    '''
        Given reciever and transmitter location, return specular point.
        Return empty element if no specular point is found.
        This is for a time series.

        Source: https://www.geometrictools.com/Documentation/SphereReflections.pdf
    '''
    global EARTH_RADIUS

    # Break down inputs
    rec = np.array([recx, recy, recz]).T / EARTH_RADIUS
    trans = np.array([transx, transy, transz]).T / EARTH_RADIUS
    
    # Prework - dot products
    a = np.einsum('ij,ij->i', rec, rec)
    b = np.einsum('ij,ij->i', rec, trans)
    c = np.einsum('ij,ij->i', trans, trans)

    # Step 1
    coeffs = np.array([4*c*(a*c-b**2), -4*(a*c-b**2), a+2*b+c-4*a*c, 2*(a-b), a-1]).T
    roots  = quartic_roots(coeffs)

    # Remove elements without positive roots
    y = roots[roots > 0].reshape((-1,2))

    # Step 2
    b = b[:, np.newaxis]
    c = c[:, np.newaxis]
    x = (-2*c*y**2 + y + 1) / (2*b*y + 1)

    # Pick x and y for which both x and y are > 0
    # Remove double samples
    double_spec = np.logical_xor(x[:,0], x[:,1])
    x[double_spec, :] = -1

    positive = x > 0
    y = y[positive][:, np.newaxis]
    x = x[positive][:, np.newaxis]

    # Remove receiver and transmitters that don't have a specular point
    spec_iloc = np.sum(positive.astype(int), axis=1).astype(bool)
    rec = rec[spec_iloc, :]
    trans = trans[spec_iloc, :]
    time = time[spec_iloc]

    try:
        spec = np.real((x*rec + y*trans) * EARTH_RADIUS)
    except ValueError:
        print(rec.shape)
        print(trans.shape)
        print(a.shape)
        print(coeffs.shape)
        print(roots.shape)
        print(y.shape)
        print(x.shape)
        print(spec_iloc.shape)
        exit()

    return spec, rec, trans, time

def get_specular_points_fuck_titan(transmitters, trans_sma, time, recivers, rec_sma, rec_satNum):
    '''
        Returns dataframe of specular points for receivers and ONE transmitter constellation.
        Specially useful for parallel programming.
        Uses this to calculate specular points: 
        https://www.geometrictools.com/Documentation/SphereReflections.pdf
    '''
    print(mp.current_process().name + ' has started working.')
    # Specular points dataframe
    spec_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'theta2', 'theta3'])

    # Vectorize the function
    vfunc = np.vectorize(get_spec)

    # Iterate thru transmitters in each shell
    for i in range(transmitters.shape[2]):
        transmitter = transmitters[:,:,i]
        # Iterate thru the recievers shells
        for j in range(len(recivers)):
            receiver_shell = recivers[j]
            # Iterate thru receivers in shell
            for k in range(rec_satNum[j]):
                reciver = np.radians(receiver_shell[:, k*2:k*2+2])

                # ECEF of transmitter and receiver
                trans_x = trans_sma * np.cos(transmitter[:,0]) * np.cos(transmitter[:,1])
                trans_y = trans_sma * np.cos(transmitter[:,0]) * np.sin(transmitter[:,1])
                trans_z = trans_sma * np.sin(transmitter[:,0])
                
                rec_x = rec_sma[j] * np.cos(reciver[:,0]) * np.cos(reciver[:,1])
                rec_y = rec_sma[j] * np.cos(reciver[:,0]) * np.sin(reciver[:,1])
                rec_z = rec_sma[j] * np.sin(reciver[:,0])
                
                # Get them goods
                spec, rec, trans, trim_time = get_spec_vectorized(rec_x, rec_y, rec_z, trans_x, trans_y, trans_z, time)
                # sp_x, sp_y, sp_z = vfunc(recx=rec_x, recy=rec_y, recz=rec_z, transx=trans_x, transy=trans_y, transz=trans_z)
                
                # Put it in a dataframe because that is easier to handle from now on
                temp_df = pd.DataFrame(columns=['Time'])
                temp_df['Time'] = trim_time/86400                    # in days

                # Add in ECEF info as it is needed later
                temp_df['spec_x'] = spec[:,0]
                temp_df['spec_y'] = spec[:,1]
                temp_df['spec_z'] = spec[:,2]
                temp_df['trans_x'] = trans[:,0]
                temp_df['trans_y'] = trans[:,1]
                temp_df['trans_z'] = trans[:,2]
                temp_df['rec_x'] = rec[:,0]
                temp_df['rec_y'] = rec[:,1]
                temp_df['rec_z'] = rec[:,2]

                temp_df = temp_df.dropna()                              # if no specular point, previous function returns none. Remove these entries

                # Finally get the LL of the specular point fuck me
                temp_df['Lat'] = np.arcsin(temp_df['spec_z'] / EARTH_RADIUS)
                temp_df['Lon'] = np.arctan2(temp_df['spec_y'], temp_df['spec_x'])
                
                # Enforce the type
                temp_df = temp_df.astype('float64')

                # Apply science requirements            
                #find r_sr, r_rt
                #r_sr
                temp_df['r_srx'] = temp_df['rec_x'] - temp_df['spec_x']
                temp_df['r_sry'] = temp_df['rec_y'] - temp_df['spec_y']
                temp_df['r_srz'] = temp_df['rec_z'] - temp_df['spec_z']

                #r_rt
                temp_df['r_rtx'] = temp_df['trans_x'] - temp_df['rec_x']
                temp_df['r_rty'] = temp_df['trans_y'] - temp_df['rec_y']
                temp_df['r_rtz'] = temp_df['trans_z'] - temp_df['rec_z']

                #find thetas (for all names to the left of '=', coefficient 'r' left out, ex: r_rt made to be rt)
                #theta1 (assumption of r_s constant?)
                temp_df['dot_s_sr'] = temp_df['r_srx']*temp_df['spec_x'] + temp_df['r_sry']*temp_df['spec_y'] + temp_df['r_srz']*temp_df['spec_z'] 
                temp_df['mag_sr'] = np.sqrt(np.square(temp_df['r_srx']) + np.square(temp_df['r_sry']) + np.square(temp_df['r_srz']))
                
                # Arcos is limited to [-1 1]. Due to numerical issues, sometimes our are slightly outside this range.
                # Force them inside the range
                temp_df['theta1_temp'] = temp_df['dot_s_sr'] / (temp_df['mag_sr']*EARTH_RADIUS)
                temp_df['theta1_temp'].where(temp_df['theta1_temp'] <= 1.0, 1.0, inplace=True)
                temp_df['theta1_temp'].where(temp_df['theta1_temp'] >= -1.0, -1.0, inplace=True)
                temp_df['theta1'] = np.abs(np.arccos(temp_df['theta1_temp'])) * 180.0 / np.pi

                #theta2
                temp_df['dot_r_sr'] = temp_df['r_srx']*temp_df['rec_x'] + temp_df['r_sry']*temp_df['rec_y'] + temp_df['r_srz']*temp_df['rec_z'] 
                temp_df['mag_r'] = np.sqrt(np.square(temp_df['rec_x']) + np.square(temp_df['rec_y']) + np.square(temp_df['rec_z']))
                temp_df['theta2'] = np.abs(np.arccos(temp_df['dot_r_sr']/(temp_df['mag_r']*temp_df['mag_sr']))) * 180.0 / np.pi

                #theta3
                temp_df['dot_rt_r'] = temp_df['r_rtx']*temp_df['rec_x'] + temp_df['r_rty']*temp_df['rec_y'] + temp_df['r_rtz']*temp_df['rec_z'] 
                temp_df['mag_rt'] = np.sqrt(np.square(temp_df['r_rtx']) + np.square(temp_df['r_rty']) + np.square(temp_df['r_rtz']))
                temp_df['theta3'] = np.abs(np.arccos(temp_df['dot_rt_r']/(temp_df['mag_r']*temp_df['mag_rt']))) * 180.0 / np.pi
                
                # Inclination angle is always < 60 deg (theta 1)
                temp_df = temp_df[temp_df['theta1'] <= 60.0]

                # Remove extra columns
                keep    = ['Time', 'Lat', 'Lon', 'theta2', 'theta3']
                extra   = [elem for elem in temp_df.columns.to_list() if elem not in keep]
                temp_df = temp_df.drop(columns=extra)

                # Transfer numpy array to list to get it to work well
                # Transform to degrees while we're at it
                temp_df['Lat'] = (temp_df['Lat']*180.0/np.pi).tolist()
                temp_df['Lon'] = (temp_df['Lon']*180.0/np.pi).tolist()

                # Append
                spec_df = pd.concat([spec_df, temp_df])
        
    print(mp.current_process().name + ' has finished working.')

    return spec_df

def get_transmitters_desired_freq(filename, trans_satNum, trans_freq, trans_sma, start, old_time, desired_freq, dt=1, days=15):
    '''
        Returns list of transmitter constellations in the desired frequency
        List is list of 3D numpy arrays. Each numpy array contains a single transmitter constellation.
        Each (:,:,i) corresponds to a transmitter in the constellation
    '''
    # Generate list of transmitter constellations
    transmitter_constellations = []
    trans_desired_sma = []

    for i in range(len(trans_satNum)):
        # Check if transmitters are in desired frequency. If not, just skip it
        if trans_freq[i] not in desired_freq:
            # Update start and then skip this transmitter constellation
            start = start + trans_satNum[i]*2
            continue

        # Get transmitters into
        transmitters = load_data(filename, columns=tuple(range(start, start+trans_satNum[i]*2)))
        start = start + trans_satNum[i]*2

        # Interpolate
        new_time, transmitters = interpolation2(old_time, transmitters, dt=dt, days=days)

        transmitters = np.radians(transmitters.reshape((new_time.shape[0],2,-1)))

        transmitter_constellations = transmitter_constellations + [transmitters]
        trans_desired_sma = trans_desired_sma + [trans_sma[i]]
    
    return transmitter_constellations, trans_desired_sma

def get_specular_points_multiprocessing(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, dt=1, days=15):
    '''
        Function which given a file with transmitters and receivers LLA, can return the specular points.
        Also does interpolation at dt.
        Built with multiprocessing in mind. You probably shouldn't run this outside of the servers...
    '''
    global EARTH_RADIUS
    print('Loading receivers & time...')

    # Get time
    old_time = load_data(filename, columns=tuple(range(0,1)))

    # Get reciever constellations.
    # List of 2D numpy arrays. Each element of the list is a shell in the MoIST constellation
    recivers = []
    time = np.array([])
    start = 1                       # stores where to start looking for satellties
    for i in range(len(rec_satNum)):
        if i == 0:
            time, temp = interpolation2(old_time, load_data(filename, columns=tuple(range(start, start+rec_satNum[i]*2))), dt=dt, days=days)
        else:
            _, temp = interpolation2(old_time, load_data(filename, columns=tuple(range(start, start+rec_satNum[i]*2))), dt=dt, days=days)
        recivers = recivers + [temp]
        start = start+rec_satNum[i]*2

    # Specular points dataframe
    spec_df_l_band = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'theta2', 'theta3'])
    spec_df_p_band = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'theta2', 'theta3'])
    spec_df_vhf_band = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'theta2', 'theta3'])

    # Get the specular points
    # Set up multiprocessing
    print('Let\'s get this party started')
    # specular_df = get_specular_points_fuck_titan(transmitter_constellations_p_band[0], trans_sma_p_band[0], time, recivers, rec_sma, rec_satNum)
    # exit()
    pool = mp.Pool()

    # L-Band
    print('Working on L-Band...')
    transmitter_constellations_l_band, trans_sma_l_band = get_transmitters_desired_freq(filename, trans_satNum, trans_freq, trans_sma, start, old_time, 'l', dt=dt, days=days)
    print('Getting specular points')
    for trans_const, trans_sma in zip(transmitter_constellations_l_band, trans_sma_l_band):
        print('\n\n')
        specular_df = get_specular_points_fuck_titan(trans_const, trans_sma, time, recivers, rec_sma, rec_satNum)
    exit()
    results_l_band = pool.starmap(partial(get_specular_points_fuck_titan, time=time, recivers=recivers, rec_sma=rec_sma, rec_satNum=rec_satNum),\
              zip(transmitter_constellations_l_band, trans_sma_l_band))
    
    # P-Band
    print('Working on P-Band...')
    transmitter_constellations_p_band, trans_sma_p_band = get_transmitters_desired_freq(filename, trans_satNum, trans_freq, trans_sma, start, old_time, 'p', dt=dt, days=days)
    results_p_band = pool.starmap(partial(get_specular_points_fuck_titan, time=time, recivers=recivers, rec_sma=rec_sma, rec_satNum=rec_satNum),\
              zip(transmitter_constellations_p_band, trans_sma_p_band))
    
    # VHF-Band
    print('Working on VHF-Band...')
    transmitter_constellations_vhf_band, trans_sma_vhf_band = get_transmitters_desired_freq(filename, trans_satNum, trans_freq, trans_sma, start, old_time, 'vhf', dt=dt, days=days)
    results_vhf_band = pool.starmap(partial(get_specular_points_fuck_titan, time=time, recivers=recivers, rec_sma=rec_sma, rec_satNum=rec_satNum),\
              zip(transmitter_constellations_vhf_band, trans_sma_vhf_band))
    pool.close()
    pool.join()

    # Join results
    if len(results_l_band) > 0:
        spec_df_l_band = pd.concat(results_l_band)
    if len(results_p_band) > 0:
        spec_df_p_band = pd.concat(results_p_band)
    if len(results_vhf_band) > 0:
        spec_df_vhf_band = pd.concat(results_vhf_band)
    
    return spec_df_l_band, spec_df_p_band, spec_df_vhf_band

def get_revisit_info(specular_df, grid):
    # Better estimate to fit points to a grid
    # Generate grid and get indeces for each specular point that matches grid element
    points = np.c_[specular_df['Lat'], specular_df['Lon']]
    tree = KDTree(grid)
    _, indices = tree.query(points)

    # Get estimated latitude and longitude based on resolution
    specular_df['approx_LatSp'] = grid[indices, 0]
    specular_df['approx_LonSp'] = grid[indices, 1]

    # Calculate time difference
    specular_df.sort_values(by=['approx_LatSp', 'approx_LonSp', 'Time'], inplace=True)
    specular_df['revisit'] = specular_df['Time'].diff()

    # Correct for borders
    specular_df['revisit'].mask(specular_df.approx_LatSp != specular_df.approx_LatSp.shift(1), other=np.nan, inplace=True)
    specular_df['revisit'].mask(specular_df.approx_LonSp != specular_df.approx_LonSp.shift(1), other=np.nan, inplace=True)

    # Get max revisit and store in new DF
    indeces = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])['revisit'].transform(max) == specular_df['revisit']

    max_rev_area_df = specular_df[indeces]

    # Get median revisit and store in new DF
    med_rev_area_series = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])['revisit'].median()
    names = med_rev_area_series.index.to_frame()

    med_rev_area_df = pd.DataFrame(columns=['approx_LatSp', 'approx_LonSp', 'revisit'])
    med_rev_area_df['approx_LatSp'] = names['approx_LatSp'].reset_index(drop=True)
    med_rev_area_df['approx_LonSp'] = names['approx_LonSp'].reset_index(drop=True)
    med_rev_area_df['revisit'] = med_rev_area_series.to_frame()['revisit'].reset_index(drop=True)

    # Any revisit that is less than 1 hour is removed. Typically this occurs because of a lack of samples (due to low sim time)
    # max_rev_area_df['revisit'].mask(max_rev_area_df['revisit'] < 0.04, other=np.nan, inplace=True)

    return max_rev_area_df, med_rev_area_df

def get_swe_100m(specular_df):
    global EARTH_RADIUS

    print('Beginning SWE 100m revisit calculations')
    # Generate grid and get indeces for each specular point that matches grid element
    specular_df = specular_df[abs(specular_df['Lat']) <= 70.0]

    grid = create_earth_grid(resolution=100, max_lat=70)
    points = np.c_[specular_df['Lat'], specular_df['Lon']]
    tree = KDTree(grid)
    _, indices = tree.query(points)

    # Get estimated latitude and longitude based on resolution
    specular_df['approx_LatSp'] = grid[indices, 0]
    specular_df['approx_LonSp'] = grid[indices, 1]

    # Calculate time difference
    # specular_df.sort_values(by=['approx_LatSp', 'approx_LonSp', 'Time'], inplace=True)
    specular_df = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])

    total_100m = 0
    total_200m = 0
    total_300m = 0
    total_400m = 0
    total_500m = 0
    total_1km  = 0
    buckets = grid.shape[0]

    with tqdm(total=specular_df.ngroups) as progress_bar:
        for _, group in specular_df:
            progress_bar.update(1)
            row_mins = []
            for index, row in group.iterrows():
                # Get distance
                distance_from_row = get_distance_lla(row['Lat'], row['Lon'], group['Lat'].drop(index), group['Lon'].drop(index))
                minimum = np.amin(distance_from_row)

                row_mins = row_mins + [minimum]
            array = np.array(row_mins)

            m_100 = array[array < 0.1]
            m_200 = array[array < 0.2]
            m_300 = array[array < 0.3]
            m_400 = array[array < 0.4]
            m_500 = array[array < 0.5]
            km_1  = array[array < 1.0]

            if m_100.size > 0:
                total_100m = total_100m + 1
            if m_200.size > 0:
                total_200m = total_200m + 1
            if m_300.size > 0:
                total_300m = total_300m + 1
            if m_400.size > 0:
                total_400m = total_400m + 1
            if m_500.size > 0:
                total_500m = total_500m + 1    
            if km_1.size > 0:
                total_1km = total_1km + 1     
    
    print('######################################################################################')
    print('Snow-Water Equivalent (SWE): L-Band Frequency')
    print('Amount of SWE 100m passes: ' + str(total_100m))
    print('Coverage of SWE 100m passes: '+ str(total_100m/buckets))

    print('Amount of SWE 200m passes: ' + str(total_200m))
    print('Coverage of SWE 200m passes: '+ str(total_200m/buckets))

    print('Amount of SWE 300m passes: ' + str(total_300m))
    print('Coverage of SWE 300m passes: '+ str(total_300m/buckets))

    print('Amount of SWE 400m passes: ' + str(total_400m))
    print('Coverage of SWE 400m passes: '+ str(total_400m/buckets))

    print('Amount of SWE 500m passes: ' + str(total_500m))
    print('Coverage of SWE 500m passes: '+ str(total_500m/buckets))

    print('Amount of SWE 1km passes: ' + str(total_1km))
    print('Coverage of SWE 1km passes: '+ str(total_1km/buckets))
    print('######################################################################################')

def get_distance_lla(row_lat, row_long, group_lat, group_long):
    def radians(degrees):
        return degrees * np.pi / 180.0
    
    global EARTH_RADIUS
    lon1 = radians(group_long)
    lon2 = radians(row_long)
    lat1 = radians(group_lat)
    lat2 = radians(row_lat)
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
 
    c = 2 * np.arcsin(np.sqrt(a))
      
    # calculate the result
    return(c * EARTH_RADIUS)

def apply_science_angles(specular_df, science_req='SSM'):
    if science_req == 'SSM' or science_req == 'FTS' or science_req == 'SWE_L':
        # this one is L-band
        specular_df = specular_df[specular_df['theta2'] <= 21.0]
        specular_df = specular_df[specular_df['theta3'] <= 62.5]
    elif science_req == 'RZSM' or science_req == 'SWE_P':
        # this one is p/VHF-band
        specular_df = specular_df[specular_df['theta2'] <= 60.0]
        specular_df = specular_df[specular_df['theta3'] <= 60.0]
    else:
        print('Not a known science requirement type')
    
    specular_df = specular_df.drop(columns=['theta2', 'theta3'])
    return specular_df

def get_revisit_stats(specular_df, science_req):
    # Apply angle requirements
    specular_df = apply_science_angles(specular_df, science_req)

    if specular_df.empty:
        print('######################################################################################')
        print('No specular points for: ' + science_req)
        print('######################################################################################')
        return

    # Get revisit stats based on science requirements
    if science_req == 'SSM' or science_req == 'RZSM':
        # Get revisit
        specular_df = specular_df[abs(specular_df['Lat']) <= 70.0]

        if science_req == 'SSM':
            grid = create_earth_grid(resolution=10, max_lat=70)
            max_revisit_info, med_revisit_info = get_revisit_info(specular_df, grid=grid)
        elif science_req == 'RZSM':
            grid = create_earth_grid(resolution=40, max_lat=70)
            max_revisit_info, med_revisit_info = get_revisit_info(specular_df, grid=grid)

        # Global
        global_max_rev = max_revisit_info[abs(max_revisit_info['approx_LatSp']) <= 50.0]
        global_med_rev = med_revisit_info[abs(med_revisit_info['approx_LatSp']) <= 50.0]
        global_buckets = np.delete(grid, np.where(abs(grid[:,0]) > 50)[0], axis=0).shape[0]

        # Calculations of Maximum revisit
        global_quantile_max = global_max_rev['revisit'].quantile(0.50)
        global_coverage_max = global_max_rev[global_max_rev['revisit'] <= global_quantile_max].shape[0] / global_buckets

        # Calculations of Median revisit
        global_quantile_med = global_med_rev['revisit'].quantile(0.99)
        global_coverage_med = global_med_rev[global_med_rev['revisit'] <= global_quantile_med].shape[0] / global_buckets

        # Coverage at 3 day revisit - Median and Max
        global_coverage_3day_max = global_max_rev[global_max_rev['revisit'] <= 3.0].shape[0] / global_buckets
        global_coverage_3day_med = global_med_rev[global_med_rev['revisit'] <= 3.0].shape[0] / global_buckets

        # Boreal
        boreal_max_rev = max_revisit_info[abs(max_revisit_info['approx_LatSp']) <= 70.0]
        boreal_max_rev = boreal_max_rev[abs(boreal_max_rev['approx_LatSp']) >= 50.0]
        boreal_med_rev = med_revisit_info[abs(med_revisit_info['approx_LatSp']) <= 70.0]
        boreal_med_rev = boreal_med_rev[abs(boreal_med_rev['approx_LatSp']) >= 50.0]
        boreal_buckets = np.delete(grid, np.where(abs(grid[:,0]) < 50)[0], axis=0).shape[0]

        # Calculations of Maximum revisit
        boreal_quantile_max = boreal_max_rev['revisit'].quantile(0.50)
        boreal_coverage_max = boreal_max_rev[boreal_max_rev['revisit'] <= boreal_quantile_max].shape[0] / boreal_buckets

        # Calculations of Median revisit
        boreal_quantile_med = boreal_med_rev['revisit'].quantile(0.99)
        boreal_coverage_med = boreal_med_rev[boreal_med_rev['revisit'] <= boreal_quantile_med].shape[0] / boreal_buckets

        # Coverage at 2 day revisit - Median and Max
        boreal_coverage_2day_max = boreal_max_rev[boreal_max_rev['revisit'] <= 2.0].shape[0] / boreal_buckets
        boreal_coverage_2day_med = boreal_med_rev[boreal_med_rev['revisit'] <= 2.0].shape[0] / boreal_buckets

        print('######################################################################################')
        print('Surface Soil Moisture (SSM): L-Band Frequency, 10km' if science_req=='SSM' else 'Root Zone Soil Moisture (RZSM):P-Band & VHF Frequency, 40km')
        print('Max latitude: ' + str(abs(specular_df['Lat']).max()))

        print('\nGlobal')
        print('50.0 Percentile of Maximum Revisit for '+science_req+' Global: ' + str(global_quantile_max))
        print('Coverage for at 50.0 Percentile of Maximum Revisit for '+science_req+' Global: ' + str(global_coverage_max))

        print('\n99.0 Percentile of Median Revisit for '+science_req+' Global: ' + str(global_quantile_med))
        print('Coverage for at 99.0 Percentile of Median Revisit for '+science_req+' Global: ' + str(global_coverage_med))

        print('\nCoverage for 3 day Maximum Revisit for '+science_req+' Global: ' + str(global_coverage_3day_max))
        print('Coverage for 3 day Median Revisit for '+science_req+' Global: ' + str(global_coverage_3day_med))

        print('\nBoreal')
        print('50.0 Percentile of Maximum Revisit for '+science_req+' Boreal: ' + str(boreal_quantile_max))
        print('Coverage for at 50.0 Percentile of Maximum Revisit for '+science_req+' Boreal: ' + str(boreal_coverage_max))

        print('\n99.0 Percentile of Median Revisit for '+science_req+' Boreal: ' + str(boreal_quantile_med))
        print('Coverage for at 99.0 Percentile of Median Revisit for '+science_req+' Boreal: ' + str(boreal_coverage_med))

        print('\nCoverage for 3 day Maximum Revisit for '+science_req+' Boreal: ' + str(boreal_coverage_2day_max))
        print('Coverage for 3 day Median Revisit for '+science_req+' Boreal: ' + str(boreal_coverage_2day_med))
        print('######################################################################################')

    elif science_req == 'FTS':
        # Get revisit
        specular_df = specular_df[abs(specular_df['Lat']) <= 60.0]
        grid = create_earth_grid(resolution=3, max_lat=60)
        max_revisit_info, med_revisit_info = get_revisit_info(specular_df, grid=grid)

        buckets = grid.shape[0]

        # Calculations of Maximum revisit
        quantile_max = max_revisit_info['revisit'].quantile(0.50)
        coverage_max = max_revisit_info[max_revisit_info['revisit'] <= quantile_max].shape[0] / buckets

        # Calculations of Median revisit
        quantile_med = med_revisit_info['revisit'].quantile(0.99)
        coverage_med = med_revisit_info[med_revisit_info['revisit'] <= quantile_med].shape[0] / buckets

        # Coverage at 3 day revisit - Median and Max
        coverage_3day_max = max_revisit_info[max_revisit_info['revisit'] <= 3.0].shape[0] / buckets
        coverage_3day_med = med_revisit_info[med_revisit_info['revisit'] <= 3.0].shape[0] / buckets

        print('######################################################################################')
        print('Freeze-Thaw State (F/T): L-Band Frequency, 3km')
        print('Max latitude: ' + str(abs(specular_df['Lat']).max()))

        print('\n50.0 Percentile of Maximum Revisit for '+science_req+': ' + str(quantile_max))
        print('Coverage for at 50.0 Percentile of Maximum Revisit for '+science_req+': ' + str(coverage_max))

        print('\n99.0 Percentile of Median Revisit for '+science_req+': ' + str(quantile_med))
        print('Coverage for at 99.0 Percentile of Median Revisit for '+science_req+': ' + str(coverage_med))

        print('\nCoverage for 3 day Maximum Revisit for '+science_req+': ' + str(coverage_3day_max))
        print('Coverage for 3 day Median Revisit for '+science_req+': ' + str(coverage_3day_med))
        print('######################################################################################')

    elif science_req == 'SWE_L':
        get_swe_100m(specular_df)
    elif science_req == 'SWE_P':
        # Get revisit
        specular_df = specular_df[abs(specular_df['Lat']) <= 60.0]
        grid = create_earth_grid(resolution=100, max_lat=60)
        max_revisit_info, med_revisit_info = get_revisit_info(specular_df, grid=grid)

        buckets = grid.shape[0]

        # Calculations of Maximum revisit
        quantile_max = max_revisit_info['revisit'].quantile(0.50)
        coverage_max = max_revisit_info[max_revisit_info['revisit'] <= quantile_max].shape[0] / buckets

        # Calculations of Median revisit
        quantile_med = med_revisit_info['revisit'].quantile(0.99)
        coverage_med = med_revisit_info[med_revisit_info['revisit'] <= quantile_med].shape[0] / buckets

        # Coverage at 15 day revisit - Median and Max
        coverage_15day_max = max_revisit_info[max_revisit_info['revisit'] <= 15.0].shape[0] / buckets
        coverage_15day_med = med_revisit_info[med_revisit_info['revisit'] <= 15.0].shape[0] / buckets

        # Outputs
        print('######################################################################################')
        print('Snow-Water Equivalent (SWE): P-Band Frequency, 100km')
        print('Max latitude: ' + str(abs(specular_df['Lat']).max()))

        print('\n50.0 Percentile of Maximum Revisit for '+science_req+': ' + str(quantile_max))
        print('Coverage for at 50.0 Percentile of Maximum Revisit for '+science_req+': ' + str(coverage_max))

        print('\n99.0 Percentile of Median Revisit for '+science_req+': ' + str(quantile_med))
        print('Coverage for at 99.0 Percentile of Median Revisit for '+science_req+': ' + str(coverage_med))

        print('\nCoverage for 15 day Maximum Revisit for '+science_req+': ' + str(coverage_15day_max))
        print('Coverage for 15 day Median Revisit for '+science_req+': ' + str(coverage_15day_med))
        print('######################################################################################')

    else:
        print('Not a known science requirement type')

def save_specular(specular_df, savefile_start, savefile_end):
    '''
        Saves a specular point dataframe with the given name and directory
    '''
    savefilename = savefile_start + savefile_end
    specular_df.to_csv(savefilename, header=None, index=None, sep=' ', mode='w')

def load_specular(filename):
    '''
        Loads specular DF that was generated using this script
    '''
    column_names = ['Time', 'Lat', 'Lon', 'theta2', 'theta3']

    try:
        data = pd.read_csv(filename, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=column_names)

    data.columns = column_names

    return data

def create_earth_grid(resolution, max_lat):
    '''
        Given a resolution in km, returns 2D array of latitude and longitude.
        Takes into account the non-square shape of grid elements.
    '''
    lat_km2deg = 110.574
    lon_km2deg = 111.320
    # Latitude is eazy peazy lemon squeezy
    dlat = resolution / lat_km2deg
    lats = np.arange(-max_lat, max_lat, step=dlat)

    grid = []
    
    # Longitude is a bit harder daddy
    for lat in lats:
        # Get longitudes at this latitude
        dlon = resolution / lon_km2deg * np.cos(np.deg2rad(lat))
        lons = np.arange(-180.0, 180.0, dlon)

        # Join them together
        same_lat = lat * np.ones_like(lons)
        temp = np.array((same_lat, lons)).T
        grid = grid + [temp]
    
    # Join the list
    grid_joined = np.concatenate(grid)
    
    return grid_joined

if __name__ == '__main__':
    # Preliminary information
    # File where the data is stored from GMAT
    filename = '/home/polfr/Documents/dummy_data/10_06_2021_GMAT/Unzipped/ReportFile1_TestforPol.txt'
    filename = '/home/polfr/Documents/dummy_data/test_data.txt'

    # Save file - used for loading and saving specular points
    savefile_start = '/home/polfr/Documents/dummy_data/specular_points_blueTeam_15day_1s_'
    savefile_start = '/home/polfr/Documents/dummy_data/10_06_2021_GMAT/Unzipped/specular_points_MUOS_3day_1s_'
    
    # Receiver information
    rec_sma = [EARTH_RADIUS+350, EARTH_RADIUS+550]
    rec_satNum = [10,6]

    # Transmitter information
    trans_satNum = [12, 12,\
                    2,\
                    13, 14,\
                    10, 10, 10,\
                    11, 11, 10, 10, 10, 10, 10,\
                    3,\
                    5,\
                    12, 11,\
                    1,\
                    12,\
                    11, 11, 10, 10, 10, 10, 10, 10, 10,\
                    3,\
                    1,\
                    12,\
                    12]
    trans_freq = ['l', 'l',\
                  'l',\
                  'l', 'l',\
                  'l', 'l', 'l',\
                  'l', 'l', 'l', 'l', 'l', 'l', 'l',\
                  'l',\
                  'p',\
                  'vhf', 'vhf',\
                  'vhf',\
                  'vhf',\
                  'vhf', 'vhf', 'vhf', 'vhf', 'vhf', 'vhf', 'vhf', 'vhf', 'vhf',\
                  'vhf',\
                  'vhf',\
                  'vhf',\
                  'vhf']
    trans_sma = [29600.11860223169, 29600.11860223169,\
                 27977.504096425982,\
                 25507.980889761526, 25507.980889761526,\
                 26560.219967218538, 26560.219967218538, 26560.219967218538,\
                 7154.894323517232, 7154.894323517232, 7154.894323517232, 7154.894323517232, 7154.894323517232, 7154.894323517232, 7154.894323517232,\
                 7032.052725011441,\
                 42164.60598791122,\
                 7159.806603357321, 7159.806603357321,\
                 7169.733155278799,\
                 7086.970861454123,\
                 6899.35020845556, 6899.35020845556, 6899.35020845556, 6899.35020845556, 6899.35020845556, 6899.35020845556, 6899.35020845556, 6899.35020845556, 6899.35020845556,\
                 6954.536583827208,\
                 6737.429588978587,\
                 6904.52413627514,\
                 6872.673000785395]

    # SMA of transmitter constellations & recivers (SMA of transmitters should be in order of appearance in GMAT)
    # rec_sma = [EARTH_RADIUS + 450]
    # trans_sma = [EARTH_RADIUS+35786, EARTH_RADIUS+35786]

    # Number of sats per constellation
    # Assumes 2 columns per sat (lat, lon); assumes our satellites go first
    # Same order as trans_sma
    # rec_satNum   = [1]
    # trans_satNum = [2,2]

    # Frequency of each transmitter constellation
    # trans_freq = ['l','p']

    # Get the specular points...
    # By recalculating
    if True:
        print('Generating specular points')
        specular_df_l_band, specular_df_p_band, specular_df_vhf_band = get_specular_points_multiprocessing(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, dt=0.5, days=3)
    # By loading them
    else:
        print('Getting specular points from file')
        specular_df_l_band = load_specular(savefile_start+'Lband.txt')
        specular_df_p_band = load_specular(savefile_start+'Pband.txt')
        specular_df_vhf_band = load_specular(savefile_start+'VHFband.txt')

    # Save the specular points
    if False:
        print('Saving specular points to file')
        save_specular(specular_df_l_band, savefile_start, 'Lband.txt')
        save_specular(specular_df_p_band, savefile_start, 'Pband.txt')
        save_specular(specular_df_vhf_band, savefile_start, 'VHFband.txt')

    print('Begin revisit calculations')
    # SSM
    desired_freq = ['l']        
    science = 'SSM'
    get_revisit_stats(specular_df_l_band, science)

    # FTS
    desired_freq = ['l']        
    science = 'FTS'
    get_revisit_stats(specular_df_l_band, science)

    # SWE P band
    desired_freq = ['p']        
    science = 'SWE_P'
    get_revisit_stats(specular_df_p_band, science)

    # RZSM
    desired_freq = ['vhf']        
    science = 'RZSM'
    specular_df = pd.concat([specular_df_p_band, specular_df_vhf_band])
    get_revisit_stats(specular_df, science)

    # SWE L band
    desired_freq = ['l']        
    science = 'SWE_L'
    get_revisit_stats(specular_df_l_band, science)
    
