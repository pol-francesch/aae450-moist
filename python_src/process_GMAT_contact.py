import numpy as np

contact_file = open('ContactLocator1_Orbit_Plane1_AWS_and_Sun.txt')
contact = contact_file.readlines()

time = []

i = 0
initial_time = contact[4][0] + contact[4][1]

for line in contact[4:-1]:
    line = line.split()
    if len(line) < 9:
        continue
    if (float(line[8]) > 120):
        time.append(float(line[8])-120.0)
        
avg_passes = len(time)/14

avg_time = np.average(time)
        
print(avg_passes, avg_time)
