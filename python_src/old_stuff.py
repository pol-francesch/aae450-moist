import pandas as pd 
import numpy as np
from python_src.OLD.Alhazen_Plotemy import branchdeducing_twofinite
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
    I'm too scared to delete these functions.
    So I'm just gonna yet them into here to avoid any issues :)
'''
EARTH_RADIUS = 6371.0

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

        # Iterate thru the recievers shells
        for j in range(len(recivers)):
            receiver_shell = recivers[j]
            # Iterate thru receivers in shell
            for k in range(rec_satNum[j]):
                reciver = np.radians(receiver_shell[:, k*2:k*2+2])

                # ECEF of transmitter and receiver
                trans_x = trans_sma[i] * np.cos(transmitters[:,0,:]) * np.cos(transmitters[:,1,:])
                trans_y = trans_sma[i] * np.cos(transmitters[:,0,:]) * np.sin(transmitters[:,1,:])
                trans_z = trans_sma[i] * np.sin(transmitters[:,0,:])
                
                rec_x = rec_sma[j] * np.cos(reciver[:,0]) * np.cos(reciver[:,1])
                rec_y = rec_sma[j] * np.cos(reciver[:,0]) * np.sin(reciver[:,1])
                rec_z = rec_sma[j] * np.sin(reciver[:,0])

                # Perform rotation that sets trans = pi/2 & other calculations
                # First rotation
                alpha = np.pi/2 - np.arctan2(trans_y, trans_x)
                beta  = np.pi/2 - np.arcsin(trans_z / trans_sma[i])
                
                trans_xp = trans_z*np.sin(beta) - trans_x*np.cos(beta)*np.sin(alpha) - trans_y*np.cos(beta)*np.cos(alpha)
                trans_yp = trans_x*np.cos(alpha) + trans_y*np.sin(alpha)

                # New axis to get 3D shite to work
                rec_x = rec_x[:, np.newaxis]
                rec_y = rec_y[:, np.newaxis]
                rec_z = rec_z[:, np.newaxis]

                rec_xp = rec_z*np.sin(beta) - rec_x*np.cos(beta)*np.sin(alpha) - rec_y*np.cos(beta)*np.cos(alpha)
                rec_yp = rec_x*np.cos(alpha) + rec_y*np.sin(alpha)
                rec_zp = rec_x*np.sin(beta)*np.sin(alpha) + rec_y*np.cos(alpha)*np.sin(beta)+rec_z*np.cos(beta)

                # Second rotation
                gamma = np.arctan2(trans_yp, trans_xp)

                rec_xpp = rec_zp*np.sin(gamma) - rec_yp*np.cos(gamma)
                rec_ypp = rec_yp*np.sin(gamma) - rec_zp*np.cos(gamma)
                rec_zpp = rec_xp
               
                # Get the rotated LLA for the receiver
                rec_latpp = np.arcsin(rec_zpp / rec_sma[j])
                rec_lonpp = np.arctan2(rec_ypp, rec_xpp)
                
                # Other calcs
                c = EARTH_RADIUS / (trans_sma[i])                   # c = R_spec / R_src
                b = EARTH_RADIUS / (rec_sma[j])                     # b = R_spec / R_obs
                
                # Get them goods
                lat_sp = vfunc(obs=rec_latpp, c=c, b=b).astype(np.float)
                lon_sp = vfunc(obs=rec_lonpp, c=c, b=b).astype(np.float)
                repeat = lat_sp.shape[1]
                # print(lat_sp)
                # Put it in a dataframe because that is easier to handle from now on
                # There may be an issue here just fyi
                # If you print the DF before you dropna, you will see some NaN for trans & rec. That doesn't make sense
                temp_df = pd.DataFrame(columns=['Time', 'trans_lat', 'trans_lon', 'rec_lat', 'rec_lon'])
                temp_df['Time'] = np.repeat(time/86400, repeat)         # in days
                temp_df['Lat_pp'] = lat_sp.reshape((-1,1))
                temp_df['Lon_pp'] = lon_sp.reshape((-1,1))
                temp_df['trans_lat'] = transmitters[:,0,:].ravel()
                temp_df['trans_lon'] = transmitters[:,1,:].ravel()
                temp_df['rec_lat'] = np.repeat(np.radians(receiver_shell[:, k*2:k*2+1]), repeat)
                temp_df['rec_lon'] = np.repeat(np.radians(receiver_shell[:, k*2+1:k*2+2]), repeat)

                # Add in rotation and ECEF info as it is needed later
                temp_df['trans_x'] = trans_x.ravel()
                temp_df['trans_y'] = trans_y.ravel()
                temp_df['trans_z'] = trans_z.ravel()
                temp_df['rec_x'] = np.repeat(rec_x.flatten(), repeat)
                temp_df['rec_y'] = np.repeat(rec_y.flatten(), repeat)
                temp_df['rec_z'] = np.repeat(rec_z.flatten(), repeat)

                temp_df['alpha'] = alpha.ravel()
                temp_df['beta'] = beta.ravel()
                temp_df['gamma'] = gamma.ravel()
                # print(temp_df)
                temp_df = temp_df.dropna()                              # if no specular point, previous function returns none. Remove these entries
                # print(temp_df)
                # Now rotate back 
                # (this is done here to avoid doing rotation on None object which can be returned from specular point)
                temp_df['spec_xpp'] = EARTH_RADIUS * np.cos(temp_df['Lon_pp']) * np.cos(temp_df['Lat_pp'])
                temp_df['spec_ypp'] = EARTH_RADIUS * np.sin(temp_df['Lon_pp']) * np.cos(temp_df['Lat_pp'])
                temp_df['spec_zpp'] = EARTH_RADIUS * np.sin(temp_df['Lat_pp'])

                temp_df['spec_xp'] = temp_df['spec_zpp']
                temp_df['spec_yp'] = -temp_df['spec_xpp']*np.cos(temp_df['gamma']) + temp_df['spec_ypp']*np.sin(temp_df['gamma'])
                temp_df['spec_zp'] = temp_df['spec_xpp']*np.sin(temp_df['gamma']) - temp_df['spec_ypp']*np.cos(temp_df['gamma'])

                temp_df['spec_x'] = -temp_df['spec_xp']*np.cos(temp_df['beta'])*np.sin(temp_df['alpha']) + temp_df['spec_yp']*np.cos(temp_df['alpha']) + temp_df['spec_zp']*np.sin(temp_df['beta'])*np.sin(temp_df['alpha'])
                temp_df['spec_y'] = -temp_df['spec_xp']*np.cos(temp_df['beta'])*np.cos(temp_df['alpha']) + temp_df['spec_yp']*np.sin(temp_df['alpha']) + temp_df['spec_zp']*np.cos(temp_df['alpha'])*np.sin(temp_df['beta'])
                temp_df['spec_z'] = temp_df['spec_xp']*np.sin(temp_df['beta']) + temp_df['spec_zp']*np.cos(temp_df['beta'])

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
                temp_df['theta1'] = 180.0 - np.abs(np.arccos(temp_df['dot_s_sr']/(temp_df['mag_sr']*EARTH_RADIUS))) * 180.0 / np.pi

                #theta2
                temp_df['dot_r_sr'] = temp_df['r_srx']*temp_df['rec_x'] + temp_df['r_sry']*temp_df['rec_y'] + temp_df['r_srz']*temp_df['rec_z'] 
                temp_df['mag_r'] = np.sqrt(np.square(temp_df['rec_x']) + np.square(temp_df['rec_y']) + np.square(temp_df['rec_z']))
                temp_df['theta2'] = np.abs(np.arccos(temp_df['dot_r_sr']/(temp_df['mag_r']*temp_df['mag_sr']))) * 180.0 / np.pi

                #theta3
                temp_df['dot_rt_r'] = temp_df['r_rtx']*temp_df['rec_x'] + temp_df['r_rty']*temp_df['rec_y'] + temp_df['r_rtz']*temp_df['rec_z'] 
                temp_df['mag_rt'] = np.sqrt(np.square(temp_df['r_rtx']) + np.square(temp_df['r_rty']) + np.square(temp_df['r_rtz']))
                temp_df['theta3'] = np.abs(np.arccos(temp_df['dot_rt_r']/(temp_df['mag_r']*temp_df['mag_rt']))) * 180.0 / np.pi
                
                # Inclination angle is always < 60 deg (theta 1)
                # temp_df = temp_df[temp_df['theta1'] <= 60.0]
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

def get_specular_points_single_trans_const(transmitters, trans_sma, time, recivers, rec_sma, rec_satNum):
    '''
        Returns dataframe of specular points for receivers and ONE transmitter constellation.
        Specially useful for parallel programming
    '''
    print(mp.current_process().name + ' has started working.')
    # Specular points dataframe
    spec_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'theta2', 'theta3'])

    # Vectorize the function
    vfunc = np.vectorize(branchdeducing_twofinite)

    # Iterate thru the recievers shells
    for j in range(len(recivers)):
        receiver_shell = recivers[j]
        # Iterate thru receivers in shell
        for k in range(rec_satNum[j]):
            reciver = np.radians(receiver_shell[:, k*2:k*2+2])

            # ECEF of transmitter and receiver
            trans_x = trans_sma * np.cos(transmitters[:,0,:]) * np.cos(transmitters[:,1,:])
            trans_y = trans_sma * np.cos(transmitters[:,0,:]) * np.sin(transmitters[:,1,:])
            trans_z = trans_sma * np.sin(transmitters[:,0,:])
            
            rec_x = rec_sma[j] * np.cos(reciver[:,0]) * np.cos(reciver[:,1])
            rec_y = rec_sma[j] * np.cos(reciver[:,0]) * np.sin(reciver[:,1])
            rec_z = rec_sma[j] * np.sin(reciver[:,0])

            # Perform rotation that sets trans = pi/2 & other calculations
            # First rotation
            alpha = np.pi/2 - np.arctan2(trans_y, trans_x)
            beta  = np.pi/2 - np.arcsin(trans_z / trans_sma)
            
            trans_xp = trans_z*np.sin(beta) - trans_x*np.cos(beta)*np.sin(alpha) - trans_y*np.cos(beta)*np.cos(alpha)
            trans_yp = trans_x*np.cos(alpha) + trans_y*np.sin(alpha)

            # New axis to get 3D shite to work
            rec_x = rec_x[:, np.newaxis]
            rec_y = rec_y[:, np.newaxis]
            rec_z = rec_z[:, np.newaxis]

            rec_xp = rec_z*np.sin(beta) - rec_x*np.cos(beta)*np.sin(alpha) - rec_y*np.cos(beta)*np.cos(alpha)
            rec_yp = rec_x*np.cos(alpha) + rec_y*np.sin(alpha)
            rec_zp = rec_x*np.sin(beta)*np.sin(alpha) + rec_y*np.cos(alpha)*np.sin(beta)+rec_z*np.cos(beta)

            # Second rotation
            gamma = np.arctan2(trans_yp, trans_xp)

            rec_xpp = rec_zp*np.sin(gamma) - rec_yp*np.cos(gamma)
            rec_ypp = rec_yp*np.sin(gamma) - rec_zp*np.cos(gamma)
            rec_zpp = rec_xp
            
            # Get the rotated LLA for the receiver
            rec_latpp = np.arcsin(rec_zpp / rec_sma[j])
            rec_lonpp = np.arctan2(rec_ypp, rec_xpp)
            
            # Other calcs
            c = EARTH_RADIUS / (trans_sma)                      # c = R_spec / R_src
            b = EARTH_RADIUS / (rec_sma[j])                     # b = R_spec / R_obs
            
            # Get them goods
            lat_sp = vfunc(obs=rec_latpp, c=c, b=b).astype(np.float64)
            lon_sp = vfunc(obs=rec_lonpp, c=c, b=b).astype(np.float64)
            repeat = lat_sp.shape[1]
            # print(lat_sp)
            # Put it in a dataframe because that is easier to handle from now on
            # There may be an issue here just fyi
            # If you print the DF before you dropna, you will see some NaN for trans & rec. That doesn't make sense
            temp_df = pd.DataFrame(columns=['Time', 'trans_lat', 'trans_lon', 'rec_lat', 'rec_lon'])
            temp_df['Time'] = np.repeat(time/86400, repeat)         # in days
            temp_df['Lat_pp'] = lat_sp.reshape((-1,1))
            temp_df['Lon_pp'] = lon_sp.reshape((-1,1))
            temp_df['trans_lat'] = transmitters[:,0,:].ravel()
            temp_df['trans_lon'] = transmitters[:,1,:].ravel()
            temp_df['rec_lat'] = np.repeat(np.radians(receiver_shell[:, k*2:k*2+1]), repeat)
            temp_df['rec_lon'] = np.repeat(np.radians(receiver_shell[:, k*2+1:k*2+2]), repeat)

            # Add in rotation and ECEF info as it is needed later
            temp_df['trans_x'] = trans_x.ravel()
            temp_df['trans_y'] = trans_y.ravel()
            temp_df['trans_z'] = trans_z.ravel()
            temp_df['rec_x'] = np.repeat(rec_x.flatten(), repeat)
            temp_df['rec_y'] = np.repeat(rec_y.flatten(), repeat)
            temp_df['rec_z'] = np.repeat(rec_z.flatten(), repeat)

            temp_df['alpha'] = alpha.ravel()
            temp_df['beta'] = beta.ravel()
            temp_df['gamma'] = gamma.ravel()
            # print(temp_df)
            temp_df = temp_df.dropna()                              # if no specular point, previous function returns none. Remove these entries
            # print(temp_df)
            # Now rotate back 
            # (this is done here to avoid doing rotation on None object which can be returned from specular point)
            temp_df['spec_xpp'] = EARTH_RADIUS * np.cos(temp_df['Lon_pp']) * np.cos(temp_df['Lat_pp'])
            temp_df['spec_ypp'] = EARTH_RADIUS * np.sin(temp_df['Lon_pp']) * np.cos(temp_df['Lat_pp'])
            temp_df['spec_zpp'] = EARTH_RADIUS * np.sin(temp_df['Lat_pp'])

            temp_df['spec_xp'] = temp_df['spec_zpp']
            temp_df['spec_yp'] = -temp_df['spec_xpp']*np.cos(temp_df['gamma']) + temp_df['spec_ypp']*np.sin(temp_df['gamma'])
            temp_df['spec_zp'] = temp_df['spec_xpp']*np.sin(temp_df['gamma']) - temp_df['spec_ypp']*np.cos(temp_df['gamma'])

            temp_df['spec_x'] = -temp_df['spec_xp']*np.cos(temp_df['beta'])*np.sin(temp_df['alpha']) + temp_df['spec_yp']*np.cos(temp_df['alpha']) + temp_df['spec_zp']*np.sin(temp_df['beta'])*np.sin(temp_df['alpha'])
            temp_df['spec_y'] = -temp_df['spec_xp']*np.cos(temp_df['beta'])*np.cos(temp_df['alpha']) + temp_df['spec_yp']*np.sin(temp_df['alpha']) + temp_df['spec_zp']*np.cos(temp_df['alpha'])*np.sin(temp_df['beta'])
            temp_df['spec_z'] = temp_df['spec_xp']*np.sin(temp_df['beta']) + temp_df['spec_zp']*np.cos(temp_df['beta'])

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
            temp_df['theta1'] = 180.0 - np.abs(np.arccos(temp_df['theta1_temp'])) * 180.0 / np.pi

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
        
    print(mp.current_process().name + ' has finished working.')

    return spec_df

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


