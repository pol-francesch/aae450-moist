from math import pi, sin, cos, radians, degrees, floor, sqrt, atan2
from datetime import datetime, timedelta
from sys import stdin
import os

from numpy.lib.function_base import average

EARTH_RADIUS = 6371.0

def MM2SMA(MM):
	MU = 398600.4418
	# ! Convert Mean motion from revs/day to rad/s
	MM = MM*((2*pi)/86400)
	return (MU/(MM**2))**(1.0/3.0)

def checksum(line):
    """Compute the TLE checksum."""
    return (sum((int(c) if c.isdigit() else c == '-') for c in line[0:-1]) % 10) == int(line[-1])

def getEccentricAnomaly(ecc, MA):
	precision = 1e-6
	iterLimit = 50
	i = 0

	EA = MA
	F = EA - ecc * sin(MA) - MA

	while ((abs(F) > precision) and (i < iterLimit)):
		EA = EA - F / (1.0 - ecc * cos(EA))
		F = EA - ecc * sin(EA) - MA
		i += 1

	return degrees(EA)

def getTrueAnomaly(ecc, EA):
	fak = sqrt(1.0 - ecc * ecc)
	return degrees(atan2(fak * sin(EA), cos(EA) - ecc))


if __name__ == '__main__':
    
    files = ['Inp_TLE_Galileo.txt','Inp_TLE_Glonass.txt','Inp_TLE_GPS.txt','Inp_TLE_Iridium.txt','Inp_TLE_MUOS.txt','Inp_TLE_ORBCOMM.txt','Inp_TLE_SWARM.txt']
    days = 15
    step_size = 15
    enable_recievers = 1
    rec_shells = 2
    num_recievers = [8, 5]
    alt = [350, 550]
    Inc = [80, 63.5]
    Ecc = 0
    RAAN = 0
    AoP = 0
    TA = [[0, 45, 90, 135, 180, 225, 270, 315],[0, 72, 144, 216, 288]]
    satNames = []
    Input_Epoch = '14 Oct 2021 00:00:00'
    Input_MJD = '29501.5'

    time = days * 24 * 3600

    if enable_recievers == 1:
        if os.path.exists("GMAT_Script_rec.script"):
            os.remove("GMAT_Script_rec.script")

        with open('GMAT_Script_rec.script', 'a') as f:
            f.write('%----------------------------------------\n'+
                    '%---------- Spacecraft\n'+
                    '%----------------------------------------\n\n')
        
        satName = 'MOIST_rec_'
        for k in range(rec_shells):
            for j in range(num_recievers[k]):
                SatName = satName + str(k) + '_' + str(j)
                satNames.append(SatName)
                SATNum = j + 1
                SMA = EARTH_RADIUS + 2 * alt[k]
                with open('GMAT_Script_rec.script', 'a') as f:
                        f.write("Create Spacecraft "+SatName+";\n" +
                            "GMAT "+SatName+".Id = '"+str(SATNum)+"';\n" +
                            "GMAT "+SatName+".DateFormat = UTCGregorian;\n" +
                            "GMAT "+SatName+".Epoch = '"+Input_Epoch+"';\n" +
                            "GMAT "+SatName+".CoordinateSystem = EarthMJ2000Eq;\n" +
                            "GMAT "+SatName+".DisplayStateType = Keplerian;\n" +
                            "GMAT "+SatName+".SMA = "+str(SMA)+";\n" +
                            "GMAT "+SatName+".ECC = "+str(Ecc)+";\n" +
                            "GMAT "+SatName+".INC = "+str(Inc[k])+";\n" +
                            "GMAT "+SatName+".RAAN = " + str(RAAN) + ";\n" +
                            "GMAT "+SatName+".AOP = " + str(AoP) + ";\n" +
                            "GMAT "+SatName+".TA = "+str(TA[k][j])+";\n\n")
        with open('GMAT_Script_rec.script', 'a') as f:
            f.write('%----------------------------------------\n'+
                    '%---------- Propagators\n'+
                    '%----------------------------------------\n\n'+
                    'Create Propagator DefaultProp;\n'+
                    'GMAT DefaultProp.FM = InternalODEModel;\n'+
                    'GMAT DefaultProp.Type = RungeKutta89;\n'+
                    'GMAT DefaultProp.InitialStepSize = 1;\n'+
                    'GMAT DefaultProp.Accuracy = 1e-07;\n'+
                    'GMAT DefaultProp.MinStep = '+str(step_size)+';\n'+
                    'GMAT DefaultProp.MaxStep = '+str(step_size)+';\n'+
                    'GMAT DefaultProp.MaxStepAttempts = 50;\n'+
                    'GMAT DefaultProp.StopIfAccuracyIsViolated = true;\n\n')  
            f.write('%----------------------------------------\n'+
                    '%---------- Subscribers\n'+
                    '%----------------------------------------\n\n'+
                    'Create ReportFile ReportFile_recievers;\n'+
                    'GMAT ReportFile_recievers.Filename = \'ReportFile_recievers.txt\';\n'+
                    'GMAT ReportFile_recievers.Precision = 10;\n'+
                    'GMAT ReportFile_recievers.Add = {'+
                    satNames[0] + '.ElapsedSecs, ')
            for name in satNames:
                f.write(name + '.Earth.Latitude, '+
                        name + '.Earth.Longitude, ')
            f.write('};\n')
            f.write('GMAT ReportFile_recievers.WriteHeaders = false;\n'+
                    'GMAT ReportFile_recievers.LeftJustify = On;\n'+
                    'GMAT ReportFile_recievers.ZeroFill = On;\n'+
                    'GMAT ReportFile_recievers.FixedWidth = true;\n'+
                    'GMAT ReportFile_recievers.Delimiter = \' \';\n'+
                    'GMAT ReportFile_recievers.ColumnWidth = 23;\n'+
                    'GMAT ReportFile_recievers.WriteReport = true;\n\n')
            
            f.write('%----------------------------------------\n'+
                    '%---------- Mission Sequence\n'+
                    '%----------------------------------------\n\n'+
                    'BeginMissionSequence;\n'+
                    'Propagate ')
            for name in satNames:
                f.write('DefaultProp('+name+') ')
            f.write('{')
            for name in satNames:
                f.write(name+'.ElapsedSecs = '+str(time)+', ')
            f.write('};\n\n')

        exit()

    if os.path.exists("GMAT_Script_trans.script"):
        os.remove("GMAT_Script_trans.script")

    with open('GMAT_Script_trans.script', 'a') as f:
        f.write('%----------------------------------------\n'+
                '%---------- Spacecraft\n'+
                '%----------------------------------------\n\n')
    
    satNames = []
    indices_for_pol = [None] * 319
    total_count = 0
    total_shells = 0

    shell_number = []
    shell_name = []
    shell_sma = []
    shell_num_sats = []

    for file in files:
        k = open('TLES/'+file)
        TLES = k.readlines()
        i = 0
        for line in TLES:
            TLES[i] = line.strip('\n')
            i += 1
        TLE_all = []
        for line in TLES:
            TLE_all.append(line)
        
        indices = [i for i in range(3, len(TLE_all) + 1) if i % 3 == 0]

        shells = 1
        shellSMA = []
        shellNumSats = []
        SatNames = []
        count = 0

        for i in indices:
            TLE = TLE_all[i-3:i][:]
            if TLE[1][:2] != '1 ' or checksum(TLE[1]) == False:
                print("Not a TLE")
                exit()
            if TLE[2][:2] != '2 'or checksum(TLE[2]) == False:
                print("Not a TLE")
                exit()

            if '-' in TLE[0]:
                TLE[0] = TLE[0].replace('-','')
            if ' ' in TLE[0]:
                TLE[0] = TLE[0].replace(' ','')
            SatName = TLE[0]
            satNames.append(TLE[0])

            (line,SAT,Desgnator,TLEEpoch,MM1,MM2,BSTAR,EType,ElementNum) = TLE[1].split()
            (line,SATNum,Inc,RAAN,Ecc,AoP,MA,MM) = TLE[2].split()[:8]
            EpochY = int(TLEEpoch[:2])
            if EpochY > 56:
                EpochY+=1900
            else:
                EpochY+=2000
            EpochD = float(TLEEpoch[2:])
            MA = float(MA)
            MM = float(MM)
            Ecc = float('0.'+Ecc)
            SMA = MM2SMA(MM)
            EA = getEccentricAnomaly(Ecc, radians(MA))
            TA = getTrueAnomaly(Ecc, radians(EA))
            Epoch = (datetime(EpochY-1,12,31) + timedelta(EpochD)).strftime("%d %b %Y %H:%M:%S.%f")[:-3]

            if count == 0:
                shellSMA.append(SMA)
                shellNumSats.append(1)
                SatNames.append(SatName)
                total_shells += 1
                indices_for_pol[total_count] = total_shells
            else:
                for i in range(shells):
                    if abs(SMA - shellSMA[i]) < 30:
                        shellNumSats[i] += 1
                        shellSMA[i] = (shellSMA[i] + SMA) / 2.0
                        indices_for_pol[total_count] = total_shells - (shells - i - 1)
                        break
                else:
                    shells += 1
                    total_shells += 1
                    indices_for_pol[total_count] = total_shells
                    shellSMA.append(SMA)
                    shellNumSats.append(1)
                    SatNames.append(SatName)
            count += 1
            total_count += 1

            with open('GMAT_Script_trans.script', 'a') as f:
                f.write("Create Spacecraft "+SatName+";\n" +
                    "GMAT "+SatName+".Id = '"+SATNum+"';\n" +
                    "GMAT "+SatName+".DateFormat = UTCGregorian;\n" +
                    "GMAT "+SatName+".Epoch = '"+Epoch+"';\n" +
                    "GMAT "+SatName+".CoordinateSystem = EarthMJ2000Eq;\n" +
                    "GMAT "+SatName+".DisplayStateType = Keplerian;\n" +
                    "GMAT "+SatName+".SMA = "+str(SMA)+";\n" +
                    "GMAT "+SatName+".ECC = "+str(Ecc)+";\n" +
                    "GMAT "+SatName+".INC = "+str(Inc)+";\n" +
                    "GMAT "+SatName+".RAAN = " + str(RAAN) + ";\n" +
                    "GMAT "+SatName+".AOP = " + str(AoP) + ";\n" +
                    "GMAT "+SatName+".TA = "+str(TA)+";\n\n")

        if shells == 1:
            """
            print('======================================')
            print('Shell #: '+str(total_shells))
            print('Name: '+SatNames[0])
            print('SMA: '+str(shellSMA[0]))
            print('Num Sats: '+str(shellNumSats[0])+'\n\n')
            """
            shell_number.append(total_shells)
            shell_name.append(SatNames[0])
            shell_sma.append(shellSMA[0])
            shell_num_sats.append(shellNumSats[0])
        else:
            for i in range(shells):
                """
                print('======================================')
                print('Shell #: '+str(total_shells - (shells - i - 1)))
                print('Name: '+SatNames[i])
                print('SMA: '+str(shellSMA[i]))
                print('Num Sats: '+str(shellNumSats[i])+'\n\n')
                """
                shell_number.append(total_shells - (shells - i - 1))
                shell_name.append(SatNames[i])
                shell_sma.append(shellSMA[i])
                shell_num_sats.append(shellNumSats[i])

        k.close()

    """
    print(shell_number)
    print(shell_name)
    print(shell_sma)
    print(shell_num_sats)
    print(indices_for_pol)
    """
    
    with open('GMAT_Script_trans.script', 'a') as f:
        f.write('%----------------------------------------\n'+
                '%---------- Propagators\n'+
                '%----------------------------------------\n\n'+
                'Create Propagator SyncProp;\n'+
                'GMAT SyncProp.FM = InternalODEModel;\n'+
                'GMAT SyncProp.Type = RungeKutta89;\n'+
                'GMAT SyncProp.InitialStepSize = 60;\n'+
                'GMAT SyncProp.Accuracy = 1e-07;\n'+
                'GMAT SyncProp.MinStep = 1;\n'+
                'GMAT SyncProp.MaxStep = 2700;\n'+
                'GMAT SyncProp.MaxStepAttempts = 50;\n'+
                'GMAT SyncProp.StopIfAccuracyIsViolated = true;\n\n'+
                'Create Propagator DefaultProp;\n'+
                'GMAT DefaultProp.FM = InternalODEModel;\n'+
                'GMAT DefaultProp.Type = RungeKutta89;\n'+
                'GMAT DefaultProp.InitialStepSize = 1;\n'+
                'GMAT DefaultProp.Accuracy = 1e-07;\n'+
                'GMAT DefaultProp.MinStep = '+str(step_size)+';\n'+
                'GMAT DefaultProp.MaxStep = '+str(step_size)+';\n'+
                'GMAT DefaultProp.MaxStepAttempts = 50;\n'+
                'GMAT DefaultProp.StopIfAccuracyIsViolated = true;\n\n')  

        f.write('%----------------------------------------\n'+
                '%---------- Subscribers\n'+
                '%----------------------------------------\n\n'+
                'Create ReportFile ReportFile_transmitters;\n'+
                'GMAT ReportFile_transmitters.Filename = \'ReportFile_transmitters.txt\';\n'+
                'GMAT ReportFile_transmitters.Precision = 10;\n'+
                'GMAT ReportFile_transmitters.Add = {'+
                satNames[0] + '.ElapsedSecs, ')
        for name in satNames:
            f.write(name + '.Earth.Latitude, '+
                    name + '.Earth.Longitude, ')
        f.write('};\n')
        f.write('GMAT ReportFile_transmitters.WriteHeaders = false;\n'+
                'GMAT ReportFile_transmitters.LeftJustify = On;\n'+
                'GMAT ReportFile_transmitters.ZeroFill = On;\n'+
                'GMAT ReportFile_transmitters.FixedWidth = true;\n'+
                'GMAT ReportFile_transmitters.Delimiter = \' \';\n'+
                'GMAT ReportFile_transmitters.ColumnWidth = 23;\n'+
                'GMAT ReportFile_transmitters.WriteReport = true;\n\n')
        
        f.write('%----------------------------------------\n'+
                '%---------- Mission Sequence\n'+
                '%----------------------------------------\n\n'+
                'Create String myString1\n'+
                'myString1 = \'Sync Up Complete\'\n\n'+
                'BeginMissionSequence;\n'+
                'Toggle ReportFile_transmitters Off\n'+
                'Propagate ')
        for name in satNames:
            f.write('SyncProp('+name+') ')
        f.write('{')
        for name in satNames:
            f.write(name+'.UTCModJulian = '+Input_MJD+', ')
        f.write('};\n')
        f.write('Write myString1 { Style = Concise, LogFile = false, MessageWindow = true };\n')
        f.write('Toggle ReportFile_transmitters On\n'
                'Propagate ')
        for name in satNames:
            f.write('DefaultProp('+name+') ')
        f.write('{')
        for name in satNames:
            f.write(name+'.ElapsedSecs = '+str(time)+', ')
        f.write('};\n\n')
