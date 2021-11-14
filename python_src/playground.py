import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fqs.fqs import quartic_roots

EARTH_RADIUS = 6371.009

# I fuck around in here
# This way we don't clog up our main files

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
    
    # Check if equation is valid
    # val_check = a*c - b**2
    # print(min(abs(val_check)))

    # Step 1
    coeffs = np.array([4*c*(a*c-b**2), -4*(a*c-b**2), a+2*b+c-4*a*c, 2*(a-b), a-1]).T
    roots  = quartic_roots(coeffs)

    # Remove elements without positive roots
    ypositive = roots > 0
    y = roots[ypositive].reshape((-1,2))
    # print(roots)
    # print(y)
    # print(rec)
    # print(trans)

    # Remove receiver and transmitters that don't have a specular point
    yspec_iloc = np.logical_or.reduce(ypositive, axis=1)

    # print(ypositive)
    # print(yspec_iloc)
    # print(yspec_iloc.shape)
    print(rec)
    print(trans)
    print(a)
    print(b)
    print(c)

    rec = rec[yspec_iloc, :]
    trans = trans[yspec_iloc, :]
    trim_time = time[yspec_iloc]
    a = a[yspec_iloc]
    b = b[yspec_iloc]
    c = c[yspec_iloc]

    # print(rec)
    # print(trans)
    # print(a)
    # print(b)
    # print(c)

    # Step 2
    b = b[:, np.newaxis]
    c = c[:, np.newaxis]

    try:
        x = (-2*c*y**2 + y + 1) / (2*b*y + 1)

        # Pick x and y for which both x and y are > 0
        # Remove double samples
        positive = (x > 0).astype(int)
        double_spec = np.logical_and(positive[:,0], positive[:,1])
        x[double_spec, :] = 0

        print(rec)
        print(trans)
        print(a)
        print(b)
        print(c)
        print(y)
        print(x)

        positive = x > 0
        y = y[positive][:, np.newaxis]
        x = x[positive][:, np.newaxis]

        # Remove receiver and transmitters that don't have a specular point
        spec_iloc = np.logical_or(positive[:,0], positive[:,1])
        rec = rec[spec_iloc, :]
        trans = trans[spec_iloc, :]
        trim_time = trim_time[spec_iloc]
        a = a[spec_iloc]
        b = b[spec_iloc]
        c = c[spec_iloc]

        print(rec)
        print(trans)
        print(a)
        print(b)
        print(c)
        print(y)
        print(x)

        spec = np.real((x*rec + y*trans) * EARTH_RADIUS)
        # print('\n')
        # print(y.shape)
        # print(x.shape)
        # print(rec.shape)
        # print(trans.shape)
        # print(spec.shape)

    except BaseException as err:
        print('in spec point vec')
        print(rec.shape)
        print(trans.shape)
        print(time.shape)
        print(trim_time.shape)
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(coeffs.shape)
        print(roots.shape)
        print(yspec_iloc.shape)
        print(spec_iloc.shape)
        print(ypositive.shape)
        print(positive.shape)
        print(double_spec.shape)
        print(roots[roots > 0].reshape((-1,2)).shape)
        print(max(np.sum(positive.astype(int), axis=1)))
        print(y.shape)
        print(x.shape)
        print(err)
        print('checking for nan elements')
        print(np.isnan(np.sum(rec)))
        print(np.isnan(np.sum(trans)))
        print(np.isnan(np.sum(y)))
        exit()

    return spec, rec, trans, trim_time

# idk mates
# receiver
rec_x = np.array([2, 0.00001, -2, 0.00001]) * EARTH_RADIUS
rec_y = np.array([0.00001, 2, 0.00001, -2]) * EARTH_RADIUS
rec_z = np.array([0.00001, 0.00001, 0.00001, 0.00001]) * EARTH_RADIUS

#transmitters
trans_x = np.array([10, 10, 10, 10]) * EARTH_RADIUS
trans_y = np.array([0.00001, 0.00001, 0.00001, 0.00001]) * EARTH_RADIUS
trans_z = np.array([0.00001, 0.00001, 0.00001, 0.00001]) * EARTH_RADIUS

# time
time = np.array([1, 2, 3, 4])

spec, recn, transn, trim_time = get_spec_vectorized(rec_x, rec_y, rec_z, trans_x, trans_y, trans_z, time)
exit()
print(spec/EARTH_RADIUS)
print(recn/EARTH_RADIUS)
print(transn/EARTH_RADIUS)

# fig = plt.figure()
# ax = plt.axes(projection='3d')

plt.plot(spec[0,0]/EARTH_RADIUS, spec[0,1]/EARTH_RADIUS, '.r')
# plt.plot(recn[0,0]/EARTH_RADIUS, recn[0,1]/EARTH_RADIUS, '.r')
plt.plot(transn[0,0]/EARTH_RADIUS, transn[0,1]/EARTH_RADIUS, '.r')
plt.show()

# ax.scatter3D(spec[0,0]/EARTH_RADIUS, spec[0,1]/EARTH_RADIUS, spec[0,2]/EARTH_RADIUS, c=trim_time[0], cmap='Dark2')
# ax.scatter3D(recn[0,0]/EARTH_RADIUS, recn[0,1]/EARTH_RADIUS, recn[0,2]/EARTH_RADIUS, c=trim_time[0], cmap='Dark2')
# ax.scatter3D(transn[0,0]/EARTH_RADIUS, transn[0,1]/EARTH_RADIUS, transn[0,2]/EARTH_RADIUS, c=trim_time[0], cmap='Dark2')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')