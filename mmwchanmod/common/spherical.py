"""
spherical.py:  Methods for spherical coordinates
"""

import numpy as np


def cart_to_sph(d):
    """
    Cartesian to spherical coordinates.

    Parameters
    ----------
    d : (n,3) array
        vector of positions

    Returns:
    -------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees
    """

    # Compute radius
    r = np.sqrt(np.sum(d ** 2, axis=1))
    r = np.maximum(r, 1e-8)

    # Compute angle of departure
    phi = np.arctan2(d[:, 1], d[:, 0]) * 180 / np.pi
    theta = np.arccos(d[:, 2] / r) * 180 / np.pi

    return r, phi, theta


def sph_to_cart(r, phi, theta):
    """
    Spherical coordinates to cartesian coordinates

    Parameters
    ----------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees

    Returns
    -------
    d : (n,3) array
        vector of positions

    """

    # Convert to radians
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180

    # Convert to cartesian
    d0 = r * np.cos(phi) * np.sin(theta)
    d1 = r * np.sin(phi) * np.sin(theta)
    d2 = r * np.cos(theta)
    d = np.stack((d0, d1, d2), axis=-1)

    return d


def spherical_add_sub(phi0, theta0, phi1, theta1, sub=True):
    """
    Angular addition and subtraction in spherical coordinates

    For addition, we start with a vector at (phi0,theta0), then rotate by
    theta1 in the (x1,x3) plance and then by phi1 in the (x1,x2) plane.
    For subtraction, we start with a vector at (phi0,theta0), then rotate by
    -phi1 in the (x1,x2) plane and then by -theta1 in the (x1,x3) plane.


    Parameters
    ----------
    phi0, theta0 : arrays of same size
        (azimuth,inclination) angle of the initial vector in degrees
    phi1, theta1 : arrays of same size
        (azimuth,inclination) angle of the rotation
    sub:  boolean
        if true, the angles are subtracted.  otherwise, they are added

    Returns
    -------
    phi2, theta2 : arrays of same size as input
        (azimuth,inclination) angle of the rotated vector

    """
    #  ^ z
    #  |
    #  |
    #  BS ----> x-axis ( this is the direction of antenna array)
    # Convert to radians
    theta0 = np.pi / 180 * theta0
    theta1 = np.pi / 180 * theta1
    phi0 = np.pi / 180 * phi0
    phi1 = np.pi / 180 * phi1

    if sub:
        # Find unit vector in direction of (theta0,phi0)
        x1 = np.sin(theta0) * np.cos(phi0) # theta is inclination angle
        x2 = np.sin(theta0) * np.sin(phi0)
        x3 = np.cos(theta0)

        y1 = x1 * np.cos(phi1) + x2 * np.sin(phi1)
        y2 = -x1 * np.sin(phi1) + x2 * np.cos(phi1)
        y3 = x3

        # Rotate by theta1 around y axis
        z1 = y1 * np.cos(theta1) - y3 * np.sin(theta1)
        z3 = y1 * np.sin(theta1) + y3 * np.cos(theta1)
        z2 = y2
        z1 = np.minimum(1, np.maximum(-1, z1))
        # Compute the angle of the transformed vector
        # we use the (z3,z2,z1) coordinate system
        phi2 = np.arctan2(z2, z3) * 180 / np.pi
        theta2 = np.arcsin(z1) * 180 / np.pi

    else:

        # Find unit vector in direction of (theta0,phi0)
        x3 = np.cos(theta0) * np.cos(phi0)
        x2 = np.cos(theta0) * np.sin(phi0)
        x1 = np.sin(theta0)

        # Rotate by theta1
        y1 = x1 * np.cos(theta1) + x3 * np.sin(theta1)
        y3 = -x1 * np.sin(theta1) + x3 * np.cos(theta1)
        y2 = x2

        # Rotate by phi1.
        z1 = y1 * np.cos(phi1) - y2 * np.sin(phi1)
        z2 = y1 * np.sin(phi1) + y2 * np.cos(phi1)
        z3 = y3
        z3 = np.minimum(1, np.maximum(-1, z3))

        # Compute angles
        phi2 = np.arctan2(z2, z1) * 180 / np.pi
        theta2 = np.arccos(z3) * 180 / np.pi
    return phi2, theta2

def rotation(phi1, theta1, x1, x2,x3):
    # counter-clock wise rotation
    # Rotate by phi1 around z axis

    y1 = x1 * np.cos(phi1) - x2 * np.sin(phi1)
    y2 = x1 * np.sin(phi1) + x2 * np.cos(phi1)
    y3 = x3

    # Rotate by theta1 around y axis
    z1 = y1 * np.cos(theta1) + y3 * np.sin(theta1)
    z3 = -y1 * np.sin(theta1) + y3 * np.cos(theta1)
    z2 = y2
    #z1 = np.minimum(1, np.maximum(-1, z1))
    return z1,z2,z3

def spherical_add_sub_new(phi0, theta0, phi1, theta1, sub = True):
    """
    Angular addition and subtraction in spherical coordinates

    For addition, we start with a vector at (phi0,theta0), then rotate by
    theta1 in the (x1,x3) plance and then by phi1 in the (x1,x2) plane.
    For subtraction, we start with a vector at (phi0,theta0), then rotate by
    -phi1 in the (x1,x2) plane and then by -theta1 in the (x1,x3) plane.


    Parameters
    ----------
    phi0, theta0 : arrays of same size
        (azimuth,inclination) angle of the initial vector in degrees
    phi1, theta1 : arrays of same size
        (azimuth,inclination) angle of the rotation
    sub:  boolean
        if true, the angles are subtracted.  otherwise, they are added

    Returns
    -------
    phi2, theta2 : arrays of same size as input
        (azimuth,elevation) angle of the rotated vector

    """

    # Convert to radians
    theta0 = np.pi / 180 * theta0
    theta1 = np.pi / 180 * theta1
    phi0 = np.pi / 180 * phi0
    phi1 = np.pi / 180 * phi1

    # Find unit vector in direction of (theta0,phi0)
    x1 = np.sin(theta0) * np.cos(phi0) # theta is inclination angle
    x2 = np.sin(theta0) * np.sin(phi0)
    x3 = np.cos(theta0)

    # we consider z,x,y axis so that antenna arrays can be aligned with z axis
    #
    #  BS ----> z-axis ( this is the direction of antenna array)
    #  |
    #  |
    #  v  x-axis
    # define new basis for x, y, and z axes
    z = [1,0,0] # z = standard basis x
    y = [0,1,0] # y = standard basis y
    x = [0,0,-1] # x = standard basis -z

    # then rotate the basis, x, y, and z by the given angles, phi1 and theta1
    # and get rotated basis x, y, and z
    rotated_basis_x = rotation(phi1, theta1, *x)
    rotated_basis_y = rotation(phi1, theta1, *y)
    rotated_basis_z = rotation(phi1, theta1, *z)

    dist_vect = np.array([x1,x2,x3]).T
    # then get each component of x, y, and z in local coordinate
    # by projecting the distance vector to rotated basis
    new_z = np.minimum(1, np.maximum(dist_vect@rotated_basis_z, -1))
    # obtain elevation angle
    theta2 = np.arcsin(new_z)*180/np.pi
    new_x = dist_vect@rotated_basis_x
    new_y = dist_vect@rotated_basis_y
    phi2 = np.arctan2(new_y, new_x)*180/np.pi
    return phi2, theta2




