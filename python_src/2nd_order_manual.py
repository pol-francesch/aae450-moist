import numpy as np 
import pandas as pd
from Alhazen_Plotemy import branchdeducing_twofinite
from tqdm import tqdm

# This is not a manual for how to do 2nd order
# This is me running 2nd order manually
EARTH_RADIUS = 6371.0

def get_specular_points(time, receiver, transmitter):

    specular_df = pd.DataFrame(columns=['Time','Lat','Lon','rec_lat','rec_lon','trans_lat', 'trans_lon'])

    for i in tqdm(range(time.shape[0])):
        # Perform transformation
        recLat = np.radians(receiver[i,0]) + np.pi/2.0 - np.radians(transmitter[i,0])
        recLon = np.radians(receiver[i,1]) + np.pi/2.0 - np.radians(transmitter[i,1])

        # Other parameters
        c = EARTH_RADIUS / (EARTH_RADIUS+35786)
        b = EARTH_RADIUS / (EARTH_RADIUS+450)

        # Get them goods
        lat = branchdeducing_twofinite(obs=recLat, c=c, b=b)
        lon = branchdeducing_twofinite(obs=recLon, c=c, b=b)

        # If a specular point is found
        if lat is not None and lon is not None:
            # Rotate the goods back
            lat = lat - np.pi + np.radians(transmitter[i,0])
            lon = lon - np.pi + np.radians(transmitter[i,1])

            # Add new row to df
            dic = {'Time': time[i], 'Lat': lat, 'Lon': lon, 'rec_lat': np.radians(receiver[i,0]),\
                'rec_lon': np.radians(receiver[i,1]), 'trans_lat': np.radians(transmitter[i,0]),\
                'trans_lon': np.radians(transmitter[i,1])}
            specular_df = specular_df.append(dic, ignore_index=True)

    return specular_df

def get_spec(rec, trans):
    '''
        Given reciever and transmitter location, return specular point.
        Return empty array if no specular point is found.
        Reciever and transmitter locations are in the order of 1.

        Source: https://www.geometrictools.com/Documentation/SphereReflections.pdf
    '''
    global EARTH_RADIUS

    # Prework - dot products
    a = np.dot(rec,rec)
    b = np.dot(rec,trans)
    c = np.dot(trans,trans)

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

    spec = x[i]*rec + y[i]*trans

    return spec*EARTH_RADIUS

def get_specular_points_2(time, receiver, transmitter):
    # SMA's
    r_sma = EARTH_RADIUS + 450
    t_sma = EARTH_RADIUS + 35786

    specular_df = pd.DataFrame(columns=['Time', 'spec_x', 'spec_y', 'spec_z', 'trans_x', 'trans_y', 'trans_z', 'rec_x', 'rec_y', 'rec_z'])

    for i in tqdm(range(time.shape[0])):
        # Transmitter ECEF 
        trans_x = t_sma * np.cos(transmitter[i,1]) * np.cos(transmitter[i,0]) / EARTH_RADIUS
        trans_y = t_sma * np.sin(transmitter[i,1]) * np.cos(transmitter[i,0]) / EARTH_RADIUS
        trans_z = t_sma * np.sin(transmitter[i,0]) / EARTH_RADIUS
        trans = np.array([trans_x, trans_y, trans_z])

        # Receiver ECEF
        rec_x = r_sma * np.cos(receiver[i,1]) * np.cos(receiver[i,0]) / EARTH_RADIUS
        rec_y = r_sma * np.sin(receiver[i,1]) * np.cos(receiver[i,0]) / EARTH_RADIUS
        rec_z = r_sma * np.sin(receiver[i,0]) / EARTH_RADIUS
        rec = np.array([rec_x, rec_y, rec_z])
        
        spec_point = get_spec(rec, trans)

        if spec_point is not None:
            dic = {'Time': time[i], 'spec_x': spec_point[0], 'spec_y': spec_point[1],\
                   'spec_z': spec_point[2], 'trans_x': trans[0]*EARTH_RADIUS,\
                   'trans_y': trans[1]*EARTH_RADIUS, 'trans_z': trans[2]*EARTH_RADIUS,\
                   'rec_x': rec[0]*EARTH_RADIUS, 'rec_y': rec[1]*EARTH_RADIUS, 'rec_z': rec[2]*EARTH_RADIUS}
            specular_df = specular_df.append(dic, ignore_index=True)
    
    print(specular_df)

    return specular_df

def apply_science_anlges(specular_df):
    # SMA's
    r_sma = EARTH_RADIUS + 450
    t_sma = EARTH_RADIUS + 35786

    # First need to find the relevant geometries
    temp_df = pd.DataFrame()

    # Transmitter ECEF
    temp_df['trans_x'] = t_sma * np.cos(specular_df['trans_lon']) * np.cos(specular_df['trans_lat'])
    temp_df['trans_y'] = t_sma * np.sin(specular_df['trans_lon']) * np.cos(specular_df['trans_lat'])
    temp_df['trans_z'] = t_sma * np.sin(specular_df['trans_lat'])

    # Receiver ECEF
    temp_df['rec_x'] = r_sma * np.cos(specular_df['rec_lon']) * np.cos(specular_df['rec_lat'])
    temp_df['rec_y'] = r_sma * np.sin(specular_df['rec_lon']) * np.cos(specular_df['rec_lat'])
    temp_df['rec_z'] = r_sma * np.sin(specular_df['rec_lat'])

    # Specular ECEF
    temp_df['spec_x'] = EARTH_RADIUS * np.cos(specular_df['Lon']) * np.cos(specular_df['Lat'])
    temp_df['spec_y'] = EARTH_RADIUS * np.sin(specular_df['Lon']) * np.cos(specular_df['Lat'])
    temp_df['spec_z'] = EARTH_RADIUS * np.sin(specular_df['Lat'])

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
    #theta (Angle between rS and rSR)
    temp_df['dot_s_sr'] = temp_df['r_srx']*temp_df['spec_x'] + temp_df['r_sry']*temp_df['spec_y'] + temp_df['r_srz']*temp_df['spec_z'] 
    temp_df['mag_sr'] = np.sqrt(np.square(temp_df['trans_x']) + np.square(temp_df['trans_y']) + np.square(temp_df['trans_z']))
    temp_df['theta1'] = np.abs(np.arccos(temp_df['dot_s_sr']/(temp_df['mag_sr']*EARTH_RADIUS))) * 180.0 / np.pi

    print(min(temp_df['theta1']))

def apply_science_angles_2(specular_df):
    # SMA's
    r_sma = EARTH_RADIUS + 450
    t_sma = EARTH_RADIUS + 35786

    # First need to find the relevant geometries
    temp_df = pd.DataFrame()
    #find r_sr, r_rt
    #r_sr
    temp_df['r_srx'] = specular_df['rec_x'] - specular_df['spec_x']
    temp_df['r_sry'] = specular_df['rec_y'] - specular_df['spec_y']
    temp_df['r_srz'] = specular_df['rec_z'] - specular_df['spec_z']

    #r_rt
    temp_df['r_rtx'] = specular_df['trans_x'] - specular_df['rec_x']
    temp_df['r_rty'] = specular_df['trans_y'] - specular_df['rec_y']
    temp_df['r_rtz'] = specular_df['trans_z'] - specular_df['rec_z']

    #find thetas (for all names to the left of '=', coefficient 'r' left out, ex: r_rt made to be rt)
    #theta (Angle between rS and rSR)
    temp_df['dot_s_sr'] = temp_df['r_srx']*specular_df['spec_x'] + temp_df['r_sry']*specular_df['spec_y'] + temp_df['r_srz']*specular_df['spec_z'] 
    temp_df['mag_sr'] = np.sqrt(np.square(specular_df['trans_x']) + np.square(specular_df['trans_y']) + np.square(specular_df['trans_z']))
    specular_df['theta1'] = np.abs(np.arccos(temp_df['dot_s_sr']/(temp_df['mag_sr']*EARTH_RADIUS))) * 180.0 / np.pi

    print(specular_df['theta1'])
    print(min(specular_df['theta1']))


def apply_science_angles_manually(specular_df):
    # SMA's
    r_sma = EARTH_RADIUS + 450
    t_sma = EARTH_RADIUS + 35786

    # First need to find the relevant geometries
    solutions = pd.DataFrame()

    for i, row in specular_df.iterrows():
        print(type(row))
        print(row)
        break

if __name__ == '__main__':
    filename = '/home/polfr/Documents/dummy_data/10_06_2021_GMAT/Unzipped/ReportFile1_TestforPol.txt'

    # Get the file
    data = np.loadtxt(filename, skiprows=1)

    # We will only compare the receiver to one transmitter
    time = data[:,0]
    receiver = data[:,1:3]
    transmitter = data[:,5:7]

    # Manually check you've got the right data
    # print(time[0])
    # print(receiver[0])
    # print(transmitter[0])

    # Get the specular points
    specular_df = get_specular_points(time, receiver, transmitter)
    print(specular_df)
    apply_science_anlges(specular_df)
    # apply_science_angles_manually(specular_df)
    # specular_df = get_specular_points_2(time, receiver, transmitter)
    # apply_science_angles_2(specular_df)