import matplotlib.pyplot as plt
import numpy as np


def plot_orbit(states, r_earth=6378.137, elevation_azimuth=None, ax=None, *args, **kwargs):
    """
    This function takes a tensor of states, and plots the orbit, together with the Earth.

    Parameters:
    ----------------
    states (``torch.tensor``): a set of len(tsince)x2x3 tensor of states,  where the first row represents the spacecraft position (in km) and the second the spacecraft velocity (in km/s). Reference frame is TEME.
    r_earth (``float``): Earth radius in km (used for the plot). Default value is 6378.137 km.
    elevation_azimuth (``tuple``): tuple of two floats, representing the elevation and azimuth angles of the plot. If None, the default values are used.
    ax (``matplotlib.axes._subplots.Axes3DSubplot``): 3D axis object. If None, a new figure is created.
    args: additional arguments to be passed to the plot function.
    kwargs: additional kwarguments to be passed to the plot function.

    Returns:
    ----------------
    ``matplotlib.axes._subplots.Axes3DSubplot``: 3D axis object
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #if ax is None, we plot the Earth
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x*r_earth, y*r_earth, z*r_earth, color='lightblue', alpha=0.3,label='Earth')
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

        # Plot latitude lines (circles)
        for phi in np.arange(0, np.pi, np.pi/6):
            x_lat = np.cos(u) * np.sin(phi)
            y_lat = np.sin(u) * np.sin(phi)
            z_lat = np.full_like(u, np.cos(phi))
            ax.plot(x_lat*r_earth, y_lat*r_earth, z_lat*r_earth, color='black',alpha=0.2)

        # Plot longitude lines
        for theta in np.arange(0, 2 * np.pi, np.pi/6):  
            x_lon = np.cos(theta) * np.sin(v)
            y_lon = np.sin(theta) * np.sin(v)
            z_lon = np.cos(v)
            ax.plot(x_lon*r_earth, y_lon*r_earth, z_lon*r_earth, color='black',alpha=0.2)
    ax.plot(states[:,0,0].numpy(), states[:,0,1].numpy(), states[:,0,2].numpy(),*args, **kwargs)
    ax.legend()
    ax.set_box_aspect([1,1,1])
    if elevation_azimuth is not None:
        ax.view_init(elev=elevation_azimuth[0], azim=elevation_azimuth[1])
    plt.tight_layout()
    return ax

def plot_tles(tles, 
              file_name=None, 
              figsize = (36,18), 
              show=True, 
              axs=None, 
              return_axs=False, 
              log_yscale=False, 
              *args, 
              **kwargs):
    """
    This function takes a list of tles as input and plots the histograms of some of their elements.
    
    Parameters:
    ----------------
    tles (``list``): list of tles, where each element is a ``dsgp4.tle.TLE`` object.
    file_name (``str``): name of the file (including path) where the plot is to be saved.
    figsize (``tuple``): figure size.
    show (``bool``): if True, the plot is shown.
    axs (``numpy.array``): array of AxesSubplot objects.
    return_axs (``bool``): if True, the function returns the array of AxesSubplot objects.
    log_yscale (``bool``): if True, the y-scale is logarithmic.
    args: additional arguments to be passed to the hist function.
    kwargs: additional kwarguments to be passed to the hist function.
    
    Returns:    
    ----------------
    ``numpy.array``: array of AxesSubplot objects
    """
    #I collect all the six variables from the TLEs:
    mean_motion, eccentricity, inclination, argument_of_perigee, raan, b_star, mean_anomaly, mean_motion_first_derivative, mean_motion_second_derivative = [], [], [], [], [], [], [], [], []
    for tle in tles:
        mean_motion.append(tle.mean_motion)
        eccentricity.append(tle.eccentricity)
        inclination.append(tle.inclination)
        argument_of_perigee.append(tle.argument_of_perigee)
        raan.append(tle.raan)
        b_star.append(tle.b_star)
        mean_anomaly.append(tle.mean_anomaly)
        mean_motion_first_derivative.append(tle.mean_motion_first_derivative)
        mean_motion_second_derivative.append(tle.mean_motion_second_derivative)

    plt.rcParams.update({'font.size': 22})
    if axs is None:
        fig, axs = plt.subplots(3, 3, figsize = figsize)

    axs[0,0].hist(mean_motion, bins = max([100,len(tles)]), *args, **kwargs)
    axs[0,0].set_xlabel('Mean Motion [rad/s]')
    x_min, x_max = min(mean_motion), max(mean_motion)
    axs[0,0].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[0,0].set_yscale('log')
    axs[0,0].grid(True)

    axs[0,1].hist(eccentricity, bins = max([100,len(tles)]), *args, **kwargs)
    axs[0,1].set_xlabel('Eccentricity [-]')
    x_min, x_max = min(eccentricity), max(eccentricity)
    axs[0,1].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[0,1].set_yscale('log')
    axs[0,1].grid(True)

    axs[0,2].hist([i*180/np.pi for i in inclination], bins = max([100,len(tles)]), *args, **kwargs)
    axs[0,2].set_xlabel('Inclination [deg]')
    x_min, x_max = min(inclination)*180/np.pi, max(inclination)*180/np.pi
    axs[0,2].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[0,2].set_yscale('log')
    axs[0,2].grid(True)

    axs[1,0].hist([omega*180/np.pi for omega in argument_of_perigee], bins = max([100,len(tles)]), *args, **kwargs)
    axs[1,0].set_xlabel('Argument of Perigee [deg]')
    x_min, x_max = min(argument_of_perigee)*180/np.pi, max(argument_of_perigee)*180/np.pi
    axs[1,0].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[1,0].set_yscale('log')
    axs[1,0].grid(True)

    axs[1,1].hist([RAAN*180/np.pi for RAAN in raan], bins = max([100,len(tles)]), *args, **kwargs)
    axs[1,1].set_xlabel('RAAN [deg]')
    x_min, x_max = min(raan)*180/np.pi, max(raan)*180/np.pi
    axs[1,1].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[1,1].set_yscale('log')
    axs[1,1].grid(True)

    axs[1,2].hist(b_star, bins = max([100,len(tles)]), *args, **kwargs)
    axs[1,2].set_xlabel('Bstar [1/m]')
    x_min, x_max = min(b_star), max(b_star)
    axs[1,2].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[1,2].set_yscale('log')
    axs[1,2].grid(True)

    axs[2,0].hist([M*180/np.pi for M in mean_anomaly], bins = max([100,len(tles)]), *args, **kwargs)
    axs[2,0].set_xlabel('Mean Anomaly [deg]')
    x_min, x_max = min(mean_anomaly)*180/np.pi, max(mean_anomaly)*180/np.pi
    axs[2,0].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[2,0].set_yscale('log')
    axs[2,0].grid(True)

    axs[2,1].hist(mean_motion_first_derivative, bins = max([100,len(tles)]), *args, **kwargs)
    axs[2,1].set_xlabel('Mean Motion 1st Der [rad/s**2]')
    x_min, x_max = min(mean_motion_first_derivative), max(mean_motion_first_derivative)
    axs[2,1].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[2,1].set_yscale('log')
    axs[2,1].grid(True)

    axs[2,2].hist(mean_motion_second_derivative, bins = max([100,len(tles)]), *args, **kwargs)
    axs[2,2].set_xlabel('Mean Motion 2nd Der [rad/s**3]')
    x_min, x_max = min(mean_motion_second_derivative), max(mean_motion_second_derivative)
    axs[2,2].set_xlim(x_min-(x_max-x_min)*0.05, x_max+(x_max-x_min)*0.05)
    if log_yscale:
        axs[2,2].set_yscale('log')
    axs[2,2].grid(True)

    if file_name is not None:
        fig.savefig(fname = file_name)
    if show and not return_axs:
        plt.show()

    if return_axs:
        return axs
    

