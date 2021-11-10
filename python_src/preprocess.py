import numpy as np
from scipy import interpolate
from tqdm import tqdm

def load_data(file_name, rows=0):
    data = np.loadtxt(file_name, skiprows=rows)

    return data

def interpolation(transmitters, dt=1, days=15, dt_in=False):
    # Generate time list (in days) of interval 1 second
    gran_time = np.linspace(0, days*24*3600, int(days*24*3600 / dt))

    if dt_in:
        time = np.linspace(0, days*24*3600-dt_in, int(days*24*3600 / dt_in)-1)
    else:
        time = transmitters[:,0]
    
    transmitters = np.delete(transmitters,0,axis=1)
    columns = transmitters.transpose()
    interpolated = [gran_time]

    for col in tqdm(columns):
        temp = interpolate.interp1d(time, col, kind='linear', fill_value='extrapolate')
        interpolated = interpolated + [temp(gran_time)]
    
    interpolated = np.array(interpolated).transpose()

    return interpolated

def reorder_transmitters(transmitters, sat_shell_assign, shell_num_sats):
    '''
        Makes sure that the columns are ordered correctly
        This way shells are together and not separated
    '''
    reordered = transmitters[:,0].reshape((transmitters.shape[0],1))                       # First thing should still be time!
    transmitters = np.delete(transmitters,0,axis=1)
    sat_shell_assign = np.array(sat_shell_assign)

    transmitters = transmitters.reshape((transmitters.shape[0],2,-1))

    for i in range(len(shell_num_sats)):
        # Get indeces for this shell
        indeces = np.argwhere(sat_shell_assign == i+1)
        indeces = indeces.flatten().tolist()

        # Assuming transmitters are in 3D
        shell = transmitters[:,:,indeces].reshape((transmitters.shape[0],-1))
        reordered = np.concatenate((reordered, shell),axis=1)
    
    return reordered

def combine_rec_trans(receivers, transmitters):
    # Ignore the transmitter times
    transmitters = np.delete(transmitters,0,axis=1)

    combined = np.concatenate((receivers, transmitters), axis=1)

    print(combined.shape)

    return combined

if __name__ == '__main__':
    shell_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    shell_type = ['GalileoE11', 'GalileoE18', 'GlonassCOSMOS2425', 'GPSPRN13', 'Iridium106', 'Iridium176',\
                'MUOS1', 'ORBCOMMFM8', 'ORBCOMMFM4', 'ORBCOMMFM109', 'SWARM4', 'SWARM7', 'SWARM9', 'SWARM21', 'SWARM87']
    shell_type = ['l','l','l','l','l','l','p','vhf','vhf','vhf','vhf','vhf','vhf','vhf','vhf']
    shell_sma = [29600.11860223169, 27977.504096425982, 25507.980889761526, 26560.219967218538, 7154.894323517232,\
                7032.052725011441, 42164.60598791122, 7159.806603357321, 7169.733155278799, 7086.970861454123, 6899.35020845556,\
                6954.536583827208, 6737.429588978587, 6904.52413627514, 6872.673000785395]
    shell_num_sats = [24, 2, 27, 30, 72, 3, 5, 23, 1, 12, 92, 3, 1, 12, 12]
    sat_shell_assign = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3,\
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,\
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,\
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,\
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 6, 5, 6, 5, 7, 7, 7, 7,\
                7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10,\
                10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 11, 11,\
                11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,\
                11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,\
                15, 15, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,\
                11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
    
    # Get recievers, interpolate
    print('Receivers')
    receivers_file = '/home/polfr/Documents/dummy_data/ReportFile_recievers_15sec_15day.txt'
    receivers = load_data(receivers_file, rows=0)
    print(receivers.shape)
    receivers = interpolation(receivers, dt=5, days=15)
    print(receivers.shape)

    # Get transmitters, reorganize them, and interpolate
    print('Transmitters')
    transmitters_file = '/home/polfr/Documents/dummy_data/ReportFile_transmitters.txt'
    transmitters = load_data(transmitters_file, rows=1)

    transmitters = reorder_transmitters(transmitters, sat_shell_assign, shell_num_sats)
    transmitters = interpolation(transmitters, dt=5, days=15, dt_in=60)

    print(transmitters.shape)

    # Combine and save files
    filename = '/home/polfr/Documents/dummy_data/15day_5s_orbit_blueTeam.txt'
    combined = combine_rec_trans(receivers, transmitters)
    np.savetxt(filename, combined)