import pandas as pd 
import numpy as np
from Alhazen_Plotemy import branchdeducing_twofinite
import matplotlib.pyplot as plt
from tqdm import tqdm

EARTH_RADIUS = 6371.0

def get_spec(rec, trans):
    '''
        Given reciever and transmitter location, return specular point.
        Return empty array if no specular point is found.
        Reciever and transmitter locations are in the order of 1.

        Source: https://www.geometrictools.com/Documentation/SphereReflections.pdf
    '''
    global EARTH_RADIUS

    # Break down the inputs
    # data = data.tolist()
    # rec = np.array(data[1:4]) / EARTH_RADIUS
    # trans = np.array(data[4:7]) / EARTH_RADIUS

    # Prework - dot products
    a = np.sum(rec*rec, axis=1)
    b = np.sum(rec*trans, axis=1)
    c = np.sum(trans*trans, axis=1)

    # Step 1
    coeffs = [4*c*(a*c-b**2), -4*(a*c-b**2), a+2*b+c-4*a*c, 2*(a-b), a-1]
    roots = np.roots(coeffs)
    y = roots[roots > 0]

    print(roots.shape)
    print(rec.shape)
    exit()
    
    if y.size == 0:
        return np.array([])
    
    # Step 2
    x = (-2*c*y**2 + y + 1) / (2*b*y + 1)

    if x[x > 0].size == 0:
        return np.array([])
    i = np.argwhere(x > 0)[0][0]

    spec = x[i]*rec + y[i]*trans

    return spec*EARTH_RADIUS

def load_data(file_name, columns=None):
    data = np.loadtxt(file_name, skiprows=0, usecols=columns)

    return data

def get_spec_rec(filename, rec_sma, trans_sma, rec_satNum, trans_satNum):
    '''
        Gets all the specular points for a receiver and all transmitters.
        This also needs to change if we change how we are loading data (which we 100% should do)
    '''
    # What I'm thinking we do is 3 loops in this function (will require more inputs) (2nd loop nested within 1st, 3rd within 2nd)
    # First loop will iterate thru sets of transmitters (ie one iteration is for SWARM, then Iridium, etc.)
    # This makes it such that at the beginning of the loop we load the transmitter data saving us some memory
    # Second loop just goes thru each transmitter in a transmitter constellation
    # Third loop goes thru the recievers in our constellation
    global EARTH_RADIUS

    # Get time and recivers
    time_rec = load_data(filename, columns=tuple(range(0,rec_satNum*2+1)))
    time = time_rec[:,0]
    rec_const = np.delete(time_rec, np.s_[0:1], axis=1)
    print('Code thinks that the reciver constellation has the following number of rows & columns: ', rec_const.shape)

    # Specular points dataframe
    spec_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon'])

    # Vectorize the function
    vfunc = np.vectorize(branchdeducing_twofinite)

    # Iterate thru the transmitter constellations
    print('Beginning to get specular points')
    counter = 1 + rec_satNum*2              # stores where to start looking for transmitters
    for k in range(len(trans_satNum)):
        # Get the data for that transmitter constellation
        # Done individually to save on memory. We will see if we need this when we move to the ECN servers
        numTrans = trans_satNum[k]
        trans_const = load_data(filename, columns=tuple(range(counter, counter+numTrans*2)))
        counter = counter + numTrans*2

        print('Code thinks that the current transmitter constellation has the following number of rows & columns: ', trans_const.shape)

        # Iterate thru transmitter constellation
        for i in tqdm(range(numTrans)):
            # Get the data for this transmitter
            trans = np.radians(trans_const[:,i*2:i*2+2])
            
            # Iterate thru receiver constellation
            for j in range(rec_satNum):
                # Get the data for this reciever
                rec = np.radians(rec_const[:,j*2:j*2+2])

                # Perform transformation that sets trans = pi/2 & other calculations
                rec = rec + np.pi/2 - trans
                c = EARTH_RADIUS / (trans_sma[k])                   # c = R_spec / R_src
                b = EARTH_RADIUS / (rec_sma)                        # b = R_spec / R_obs

                # Get them goods
                lat_sp = vfunc(obs=rec[:,0], c=c, b=b)
                lon_sp = vfunc(obs=rec[:,1], c=c, b=b)

                # Temp DF
                temp_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'trans_lat', 'trans_lon'])
                temp_df['Time'] = time / 86400              # in days
                temp_df['Lat']  = lat_sp
                temp_df['Lon']  = lon_sp
                temp_df['trans_lat'] = trans[:,0]
                temp_df['trans_lon'] = trans[:,1]
                temp_df['rec_lat'] = rec_const[:,j*2:j*2+1]
                temp_df['rec_lon'] = rec_const[:,j*2+1:j*2+2]
                temp_df = temp_df.dropna()                  # if no specular point, it returns none

                # Now rotate back 
                # (this is done here to avoid doing rotation on None object which can be returned from specular point)
                temp_df['Lat'] = (temp_df['Lat'] - np.pi/2 + temp_df['trans_lat'])*180/np.pi
                temp_df['Lon'] = (temp_df['Lon'] - np.pi/2 + temp_df['trans_lon'])*180/np.pi

                # Bring transmitter back to deg
                temp_df['trans_lat'] = temp_df['trans_lat']*180/np.pi
                temp_df['trans_lon'] =  temp_df['trans_lon']*180/np.pi

                # Apply science requirements
                # Get ECEF for rec, trans, spec
                temp_df['trans_x'] = trans_sma[k] * np.cos(temp_df['trans_lon']) * np.cos(temp_df['trans_lat'])
                temp_df['trans_y'] = trans_sma[k] * np.sin(temp_df['trans_lon']) * np.cos(temp_df['trans_lat'])
                temp_df['trans_z'] = trans_sma[k] * np.sin(temp_df['trans_lat'])
                
                #receiv. r vector components
                temp_df['rec_x'] = rec_sma * np.cos(temp_df['rec_lon']) * np.cos(temp_df['rec_lat'])
                temp_df['rec_y'] = rec_sma * np.sin(temp_df['rec_lon']) * np.cos(temp_df['rec_lat'])
                temp_df['rec_z'] = rec_sma * np.sin(temp_df['rec_lat'])

                # specular point, r vector components
                temp_df['spec_x'] = EARTH_RADIUS * np.cos(temp_df['Lon']) * np.cos(temp_df['Lat'])
                temp_df['spec_y'] = EARTH_RADIUS * np.sin(temp_df['Lon']) * np.cos(temp_df['Lat'])
                temp_df['spec_z'] = EARTH_RADIUS * np.sin(temp_df['Lat'])
            
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
                temp_df['mag_sr'] = np.sqrt(np.square(temp_df['trans_x']) + np.square(temp_df['trans_y']) + np.square(temp_df['trans_z']))
                temp_df['theta1'] = np.abs(np.arccos(temp_df['dot_s_sr']/(temp_df['mag_sr']*EARTH_RADIUS))) * 180.0 / np.pi

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
                temp_df['Lat'] = temp_df['Lat'].tolist()
                temp_df['Lon'] = temp_df['Lon'].tolist()

                # Append
                spec_df = pd.concat([spec_df, temp_df])
    
    return spec_df

def get_specular_points(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, desired_freq):
    '''
        This does the same as get_spec_rec but faster.        
        Gets the LL of specular points given the LL of transmitters and recievers
        LL of transmitters and recievers is in the filename
    '''
    global EARTH_RADIUS

    # Get time
    time = load_data(filename, columns=tuple(range(0,1)))

    # Get reciever constellations.
    # List of 2D numpy arrays. Each element of the list is a shell in the MoIST constellation
    recivers = []
    start = 1                       # stores where to start looking for satellties
    for i in range(len(rec_satNum)):
        recivers = recivers + [load_data(filename, columns=tuple(range(start, start+rec_satNum[i]*2)))]
        start = start+rec_satNum[i]*2
    
    # Specular points dataframe
    spec_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'theta2', 'theta3'])

    # Vectorize the function
    vfunc = np.vectorize(branchdeducing_twofinite)

    # Iterate thru the transmitter constellations
    print('Beginning to get specular points')
    for i in tqdm(range(len(trans_satNum))):
        # Check if transmitters are in desired frequency. If not, just skip it
        if trans_freq[i] not in desired_freq:
            continue

        # Get transmitters into 3D numpy array. [:,:,i] where i refers to each satellite in transmitter constellation
        transmitters = load_data(filename, columns=tuple(range(start, start+trans_satNum[i]*2)))
        start = start + trans_satNum[i]*2
        transmitters = np.radians(transmitters.reshape((time.shape[0],2,-1)))
        # print('Code thinks that the current transmitter constellation has the following shape: ', transmitters.shape)

        # Iterate thru the recievers shells
        for j in range(len(recivers)):
            receiver_shell = recivers[j]
            # Iterate thru receivers in shell
            for k in range(rec_satNum[j]):
                reciver = np.radians(receiver_shell[:, k*2:k*2+2])

                # Perform transformation that sets trans = pi/2 & other calculations
                reciver = reciver + np.pi/2
                reciver = np.subtract(reciver[...,np.newaxis], transmitters)

                # Other calcs
                c = EARTH_RADIUS / (trans_sma[i])                   # c = R_spec / R_src
                b = EARTH_RADIUS / (rec_sma[j])                     # b = R_spec / R_obs
                
                # Get them goods
                lat_sp = vfunc(obs=reciver[:,0,:], c=c, b=b).astype(np.float)
                lon_sp = vfunc(obs=reciver[:,1,:], c=c, b=b).astype(np.float)
                repeat = lat_sp.shape[1]

                # Put it in a dataframe because that is easier to handle from now on
                # There may be an issue here just fyi
                # If you print the DF before you dropna, you will see some NaN for trans & rec. That doesn't make sense
                temp_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'trans_lat', 'trans_lon', 'rec_lat', 'rec_lon'])
                temp_df['Time'] = np.repeat(time/86400, repeat)         # in days
                temp_df['Lat'] = lat_sp.reshape((-1,1))
                temp_df['Lon'] = lon_sp.reshape((-1,1))
                temp_df['trans_lat'] = transmitters[:,0,:].ravel()
                temp_df['trans_lon'] = transmitters[:,1,:].ravel()
                temp_df['rec_lat'] = np.repeat(np.radians(receiver_shell[:, k*2:k*2+1]), repeat)
                temp_df['rec_lon'] = np.repeat(np.radians(receiver_shell[:, k*2+1:k*2+2]), repeat)
                temp_df = temp_df.dropna()                              # if no specular point, previous function returns none. Remove these entries
                
                # Now rotate back 
                # (this is done here to avoid doing rotation on None object which can be returned from specular point)
                temp_df['Lat'] = temp_df['Lat'] - np.pi/2 + temp_df['trans_lat']
                temp_df['Lon'] = temp_df['Lon'] - np.pi/2 + temp_df['trans_lon']

                # Enforce the type
                temp_df = temp_df.astype('float64')

                # Apply science requirements
                # Get ECEF for rec, trans, spec
                #trans. r vector components
                temp_df['trans_x'] = trans_sma[i] * np.cos(temp_df['trans_lon']) * np.cos(temp_df['trans_lat'])
                temp_df['trans_y'] = trans_sma[i] * np.sin(temp_df['trans_lon']) * np.cos(temp_df['trans_lat'])
                temp_df['trans_z'] = trans_sma[i] * np.sin(temp_df['trans_lat'])
               
                #receiv. r vector components
                temp_df['rec_x'] = rec_sma[j] * np.cos(temp_df['rec_lon']) * np.cos(temp_df['rec_lat'])
                temp_df['rec_y'] = rec_sma[j] * np.sin(temp_df['rec_lon']) * np.cos(temp_df['rec_lat'])
                temp_df['rec_z'] = rec_sma[j] * np.sin(temp_df['rec_lat'])

                # specular point, r vector components
                temp_df['spec_x'] = EARTH_RADIUS * np.cos(temp_df['Lon']) * np.cos(temp_df['Lat'])
                temp_df['spec_y'] = EARTH_RADIUS * np.sin(temp_df['Lon']) * np.cos(temp_df['Lat'])
                temp_df['spec_z'] = EARTH_RADIUS * np.sin(temp_df['Lat'])
            
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
                temp_df['mag_sr'] = np.sqrt(np.square(temp_df['trans_x']) + np.square(temp_df['trans_y']) + np.square(temp_df['trans_z']))
                temp_df['theta1'] = np.abs(np.arccos(temp_df['dot_s_sr']/(temp_df['mag_sr']*EARTH_RADIUS))) * 180.0 / np.pi

                #theta2
                temp_df['dot_r_sr'] = temp_df['r_srx']*temp_df['rec_x'] + temp_df['r_sry']*temp_df['rec_y'] + temp_df['r_srz']*temp_df['rec_z'] 
                temp_df['mag_r'] = np.sqrt(np.square(temp_df['rec_x']) + np.square(temp_df['rec_y']) + np.square(temp_df['rec_z']))
                temp_df['theta2'] = np.abs(np.arccos(temp_df['dot_r_sr']/(temp_df['mag_r']*temp_df['mag_sr']))) * 180.0 / np.pi

                #theta3
                temp_df['dot_rt_r'] = temp_df['r_rtx']*temp_df['rec_x'] + temp_df['r_rty']*temp_df['rec_y'] + temp_df['r_rtz']*temp_df['rec_z'] 
                temp_df['mag_rt'] = np.sqrt(np.square(temp_df['r_rtx']) + np.square(temp_df['r_rty']) + np.square(temp_df['r_rtz']))
                temp_df['theta3'] = np.abs(np.arccos(temp_df['dot_rt_r']/(temp_df['mag_r']*temp_df['mag_rt']))) * 180.0 / np.pi
                
                # Inclination angle is always < 60 deg (theta 1)
                # print(temp_df)
                temp_df = temp_df[temp_df['theta1'] <= 60.0]
                # print(temp_df)

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
            
    return spec_df

def get_revisit_info(specular_df):
    print('Beginning revisit calculations')
    # Round lat and long and then use groupby to throw them all in similar buckets
    specular_df['approx_LatSp'] = round(specular_df['Lat'],1)
    specular_df['approx_LonSp'] = round(specular_df['Lon'],1)

    # Calculate time difference
    specular_df.sort_values(by=['approx_LatSp', 'approx_LonSp', 'Time'], inplace=True)
    specular_df['revisit'] = specular_df['Time'].diff()

    # Correct for borders
    specular_df['revisit'].mask(specular_df.approx_LatSp != specular_df.approx_LatSp.shift(1), other=np.nan, inplace=True)
    specular_df['revisit'].mask(specular_df.approx_LonSp != specular_df.approx_LonSp.shift(1), other=np.nan, inplace=True)

    # Get max revisit and store in new DF
    indeces = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])['revisit'].transform(max) == specular_df['revisit']

    max_rev_area_df = specular_df[indeces]

    # Any revisit that is less than 1 hour is removed. Typically this occurs because of a lack of samples (due to low sim time)
    max_rev_area_df['revisit'].mask(max_rev_area_df['revisit'] < 0.04, other=np.nan, inplace=True)

    return max_rev_area_df

def plot_revisit_stats(revisit_info):
    print('Beginning plotting')
    # Remove NaNs just to make sure
    max_rev_area_df = revisit_info[revisit_info['revisit'].notnull()]

    # Plot over all areas
    print('Creating histogram')
    ax = max_rev_area_df['revisit'].plot.hist(bins=50, alpha=0.5)
    ax.plot()
    plt.xlabel('Maximum Revisit Time (days)')
    plt.title('Maximum Revisit Frequency Distribution \n MUOS Constellation w/ 1 Satellite \n Simulation: 1s, 3 days, 1$^\circ$x1$^\circ$ Grid')
    plt.show()

def lla_to_cart(latitude, longitude):     
    """
    Convert from latitude and longitude values to cartesian vector 
    for a specular point on spherical earth
    """
    cart = []
    R = EARTH_RADIUS   
    cart.append(R * np.cos(longitude) * np.cos(latitude))
    cart.append(R * np.sin(longitude) * np.cos(latitude))
    cart.append(R * np.sin(latitude))
    return cart

def get_swe_100m(specular_df):
    global EARTH_RADIUS

    print('Beginning SWE 100m revisit calculations')
    # Round lat and long to 1 deg
    specular_df['approx_LatSp'] = round(specular_df['Lat'])
    specular_df['approx_LonSp'] = round(specular_df['Lon'])

    # Calculate time difference
    # specular_df.sort_values(by=['approx_LatSp', 'approx_LonSp', 'Time'], inplace=True)
    specular_df = specular_df.groupby(['approx_LatSp', 'approx_LonSp'])

    for name, group in specular_df:
        # print(group)
        test = group.apply(lambda row: get_distance_lla(row['Lat'], row['Lon'], group['Lat'], group['Lon']), axis=1)
        test = test.to_numpy().reshape((1,-1))
        
        km_1 = test[test < 1.0]

        if len(km_1) > 0:
            print('For the following group: ' + str(name) + ' the following points exist: ' + str(km_1))

def get_distance_lla(row_lat, row_long, group_lat, group_long):
    def radians(degrees):
        return degrees * np.pi / 180.0
    
    global EARTH_RADIUS
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(row_lat)
    lon2 = radians(row_long)
    lat1 = radians(group_lat)
    lat2 = radians(group_long)
      
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
        specular_df = specular_df[specular_df['theta2'] < 21.0]
        specular_df = specular_df[specular_df['theta3'] < 62.5]
    elif science_req == 'RZSM' or science_req == 'SWE_P':
        # this one is p/VHF-band
        specular_df = specular_df[specular_df['theta2'] < 60.0]
        specular_df = specular_df[specular_df['theta3'] < 60.0]
    else:
        exit('Not a known science requirement type')
    
    specular_df = specular_df.drop(columns=['theta2', 'theta3'])
    return specular_df

def get_revisit_stats(specular_df, science_req):
    # Apply angle requirements
    specular_df = apply_science_angles(specular_df, science_req)
    # Get revisit
    revisit_info = get_revisit_info(specular_df)

    # Get revisit stats based on science requirements
    if science_req == 'SSM' or science_req == 'RZSM':
        # Global
        global_rev = revisit_info[revisit_info['approx_LatSp'] <= 50.0]

        print('99.0 Percentile of Maximum Revisit for '+science_req+' Global: ' + str(global_rev['revisit'].quantile(0.90)))
        print('99.9 Percentile of Maximum Revisit for '+science_req+' Global: ' + str(global_rev['revisit'].quantile(0.99)))

        # Boreal forest
        boreal = revisit_info[revisit_info['approx_LatSp'] <= 70.0]
        boreal = revisit_info[revisit_info['approx_LatSp'] >= 50.0]

        print('99.0 Percentile of Maximum Revisit for '+science_req+' Boreal: ' + str(global_rev['revisit'].quantile(0.90)))
        print('99.9 Percentile of Maximum Revisit for '+science_req+' Boreal: ' + str(global_rev['revisit'].quantile(0.99)))

    elif science_req == 'FTS':
        # Apply latitudes
        revisit_info = revisit_info[revisit_info['approx_LatSp'] <= 60.0]
        
        # Show results
        print('99.0 Percentile of Maximum Revisit for FTS: ' + str(revisit_info['revisit'].quantile(0.90)))
        print('99.9 Percentile of Maximum Revisit for FTS: ' + str(revisit_info['revisit'].quantile(0.99)))
    elif science_req == 'SWE_L':
        print('TODO: SWE_L')
    elif science_req == 'SWE_P':
        # Apply latitudes
        revisit_info = revisit_info[revisit_info['approx_LatSp'] <= 60.0]
        
        # Show results
        print('99.0 Percentile of Maximum Revisit for SWE P-Band: ' + str(revisit_info['revisit'].quantile(0.90)))
        print('99.9 Percentile of Maximum Revisit for SWE P-Band: ' + str(revisit_info['revisit'].quantile(0.99)))
    else:
        exit('Not a known science requirement type')

if __name__ == '__main__':
    # Preliminary information
    # File where the data is stored from GMAT
    filename = '/home/polfr/Downloads/15day_2orbit_blueTeam.txt'
    filename = '/home/polfr/Documents/dummy_data/10_18_2021_GMAT/15day_15s_2orbit_blueTeam.txt'

    #Simons file path
    # filename = '/Users/michael/Desktop/ReportFile1_TestforPol.txt'
    
    # Receiver information
    rec_sma = [EARTH_RADIUS+350, EARTH_RADIUS+550]
    rec_satNum = [6,6]

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

    # # Number of sats per constellation
    # # Assumes 2 columns per sat (lat, lon); assumes our satellites go first
    # # Same order as trans_sma
    # rec_satNum   = [1]
    # trans_satNum = [2,2]

    # # Frequency of each transmitter constellation
    # trans_freq = ['p','p']

    # SSM
    desired_freq = ['l']        
    science = 'SSM'
    specular_df = get_specular_points(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, desired_freq)
    get_revisit_stats(specular_df, science)

    # FTS
    desired_freq = ['l']        
    science = 'FTS'
    # specular_df = get_specular_points(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, desired_freq)
    get_revisit_stats(specular_df, science)

    # RZSM
    desired_freq = ['p', 'vhf']        
    science = 'RZSM'
    specular_df = get_specular_points(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, desired_freq)
    get_revisit_stats(specular_df, science)

    # SWE P band
    desired_freq = ['p']        
    science = 'SWE_P'
    specular_df = get_specular_points(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, desired_freq)
    get_revisit_stats(specular_df, science)

    # SWE L band
    desired_freq = ['l']        
    science = 'SWE_L'
    # specular_df = get_specular_points(filename, rec_sma, trans_sma, rec_satNum, trans_satNum, trans_freq, desired_freq)
    # get_revisit_stats(specular_df, science)
    
