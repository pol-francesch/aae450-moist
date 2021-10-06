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

def load_data(file_name):
    '''
        This is not how I would like to do this in the future.
        We should split up the transmitters into different files whilst making sure that they are in the same time step.
        Right now, for 6 sats over 3 days its 128 MB. This probably scales linearly so for >100 sats and 15 days we are 
        kinda fucked.
    '''
    # data = pd.read_csv(file_name, delim_whitespace=True)
    data = np.loadtxt(file_name, skiprows=1)

    return data

def get_spec_rec(data):
    '''
        Gets all the specular points for a receiver and all transmitters.
        This also needs to change if we change how we are loading data (which we 100% should do)
    '''
    # What I'm thinking we do is 3 loops in this function (will require more inputs) (2nd loop nested within 1st, 3rd within 2nd)
    # First loop will iterate thru sets of transmitters (ie one iteration is for SWARM, then Iridium, etc.)
    # This makes it such that at the beginning of the loop we load the transmitter data saving us some memory
    # Second loop just goes thru each transmitter in a transmitter constellation
    # Third loop goes thru the recievers in our constellation (for now, only 2nd is done)
    global EARTH_RADIUS

    # For now, it is simple for me to check how many transmitters are in a constellation
    num_trans = int((data.shape[1] - 3) / 2)
    print('This code thinks there are this many transmitters: ', num_trans)

    # Reduce the reciever to a single column which has the vector information
    # This would be done for each receiver individually
    rec = np.radians(data[:,1:3])

    time = data[:,0]

    # Specular points dataframe
    spec_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon'])

    # Vectorize the function
    vfunc = np.vectorize(branchdeducing_twofinite)
    
    for i in tqdm(range(num_trans)):
        # Get the column names that pertain for this transmitter
        trans = np.radians(data[:,i*2+3:i*2+5])

        # Perform transformation that sets trans = pi/2 & other calculations
        rec1 = rec + np.pi/2 - trans
        c = EARTH_RADIUS / (EARTH_RADIUS + 35786)               # c = R_spec / R_src
        b = EARTH_RADIUS / (EARTH_RADIUS + 450)                 # b = R_spec / R_obs

        # Get them goods
        lat_sp = vfunc(obs=rec1[:,0], c=c, b=b)
        lon_sp = vfunc(obs=rec1[:,1], c=c, b=b)

        # Temp DF
        temp_df = pd.DataFrame(columns=['Time', 'Lat', 'Lon', 'trans_lat', 'trans_lon'])
        temp_df['Time'] = time / 86400              # in days
        temp_df['Lat']  = lat_sp
        temp_df['Lon']  = lon_sp
        temp_df['trans_lat'] = trans[:,0]
        temp_df['trans_lon'] = trans[:,1]
        temp_df = temp_df.dropna()

        # Now rotate back (this is to avoid doing rotation on None object which can be returned from specular point)
        temp_df['Lat'] = (temp_df['Lat'] - np.pi/2 + temp_df['trans_lat'])*180/np.pi
        temp_df['Lon'] = (temp_df['Lon'] - np.pi/2 + temp_df['trans_lon'])*180/np.pi

        # and get rid of transmitter because we don't need that anymore
        temp_df = temp_df.drop(columns=['trans_lat', 'trans_lon'])

        # Transfer numpy array to list to get it to work well
        temp_df['Lat'] = temp_df['Lat'].tolist()
        temp_df['Lon'] = temp_df['Lon'].tolist()

        # Append
        spec_df = pd.concat([spec_df, temp_df])
    
    print(spec_df.dtypes)
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
    # Remove NaNs
    max_rev_area_df = revisit_info[revisit_info['revisit'].notnull()]

    # Plot over all areas
    print('Creating histogram')
    print(max_rev_area_df)
    ax = max_rev_area_df['revisit'].plot.hist(bins=50, alpha=0.5)
    ax.plot()
    plt.xlabel('Maximum Revisit Time (days)')
    plt.title('Maximum Revisit Frequency Distribution \n MUOS Constellation w/ 1 Satellite \n Simulation: 1s, 3 days')
    plt.show()


if __name__ == '__main__':
    filename = '/home/polfr/Documents/dummy_data/10_06_2021_GMAT/Unzipped/ReportFile1_TestforPol.txt'
    data = load_data(filename)
    specular_df = get_spec_rec(data)
    revisit_info = get_revisit_info(specular_df)
    plot_revisit_stats(revisit_info)