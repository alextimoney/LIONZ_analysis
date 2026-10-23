import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

# ==============================================================================
# DEFINE SYSTEM PARAMETERS
# ==============================================================================
wavelength_um = 0.532 # 0.532 microns
m = 1 # magnification = 1
n_c = 1.1e21/(wavelength_um**2)
L = 416 # 4160 mm to first lens
dens_corr = (2*m*n_c)/L

# ==============================================================================
# GENERAL FITTING FUNCTIONS
# ==============================================================================

def gaussian(x, amplitude, mean, std_dev, offset):
    '''
    gaussian fit
    '''
    return -amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) + offset
def double_gaussian(x, x1, w1, A1, x2, w2, A2):
    '''
    negative and positive peak gaussian
    '''
    peak_neg = A1 * np.exp(-((x - x1) / w1)**2)
    peak_pos = A2 * np.exp(-((x - x2) / w2)**2)
    return peak_neg + peak_pos
def lorentzian(x, x0, gamma, amp, baseline):
    """
    x: independent variable
    x0: center of the peak
    gamma: Full Width at Half Maximum (FWHM)
    amp: amplitude (height above baseline)
    baseline: vertical offset
    """
    return baseline + amp * (gamma**2 / ((x - x0)**2 + gamma**2))

# ==============================================================================
# CALIBRATION FUNCTIONS
# ==============================================================================

def get_min_brightness_gauss(x_data, y_data, start_index, end_index, plot = True):
    '''
    Fits gaussian to the data to find the minimum brightness and its position
    '''
    mask = (x_data >= start_index) & (x_data <= end_index)
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    if len(x_fit) == 0:
        raise ValueError("The provided range contains no data points.")

    guess_amp = np.min(y_fit)-np.max(y_fit)
    guess_center = x_fit[np.argmin(y_fit)]
    guess_sigma = 10.0 
    guess_offset = np.max(y_fit)

    initial_guess = [guess_amp, guess_center, guess_sigma, guess_offset]
    popt, cov = curve_fit(gaussian, x_fit, y_fit, p0=initial_guess, maxfev=100000)
    amp, center, sigma, offset = popt

    min_brightness = amp + offset
    min_position = center

    if plot:
        x_detailed = np.linspace(x_fit.min(), x_fit.max(), 500)
        y_detailed = gaussian(x_detailed, *popt) # Evaluated ONLY once
        
        plt.plot(x_detailed, y_detailed, linewidth=2, 
                label=f'Gaussian Fit {start_index}-{end_index}', color=cm.batlowS(0))

    return min_brightness, min_position

def get_min_brightness_lorentz(x_data, y_data, start_index, end_index, plot = True):
    '''
    Fits lorentzian to the data to find the minimum brightness and its position
    '''
    mask = (x_data >= start_index) & (x_data <= end_index)
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    if len(x_fit) == 0:
        raise ValueError("The provided range contains no data points.")

    y_mean = np.mean(y_fit)
    y_min = np.min(y_fit)
    y_max = np.max(y_fit)

    if abs(y_max - y_mean) > abs(y_mean - y_min):
        # We are fitting a PEAK
        guess_amp = y_max - y_min
        guess_center = x_fit[np.argmax(y_fit)]
        guess_offset = y_min
    else:
        # We are fitting a DIP
        guess_amp = y_min - y_max
        guess_center = x_fit[np.argmin(y_fit)]
        guess_offset = y_max
    guess_gamma = (x_fit.max() - x_fit.min()) * 0.1

    initial_guess = [guess_center, guess_gamma, guess_amp, guess_offset]
    popt, cov = curve_fit(lorentzian, x_fit, y_fit, p0=initial_guess, maxfev=100000)
    center, gamma, amp, offset = popt
    extrema_brightness = amp + offset
    extrema_position = center

    if plot:
        x_detailed = np.linspace(x_fit.min(), x_fit.max(), 500)
        y_detailed = lorentzian(x_detailed, *popt) # Evaluated ONLY once
        
        plt.plot(x_detailed, y_detailed, linewidth=2, 
                label=f'Lorentzian Fit {start_index}-{end_index}', color=cm.batlowS(0))

    return extrema_brightness, extrema_position


## !! The following three functions have yet to be tested in this form !! ##

def calculate_object_edges(x, y, row_idx, x_range=None, show_plot=False, object_type = 'dark'):
    """
    Calculates the width of a dark region in a 1D intensity lineout using first derivative analysis
    Note: this function often requires a visual check, since the derivative method can be sensitive to 
          noise and may not always yield accurate results.
    Right now only is in the x direction but could be adapted to y direction if needed.
    """
    # smooth data using savgol filter to reduce noise
    y_smooth = savgol_filter(y, window_length=51, polyorder=3)
    
    # create ROI if x_range provided 
    mid = 1607
    if x_range is not None:
        mask = (x >= x_range[0]) & (x <= x_range[1])
    else:
        mask = np.ones_like(x, dtype=bool) # Use all data if no range given
        
    x_roi = x[mask]
    y_roi = y_smooth[mask]

    # ROI check: ensure that the ROI contains data points
    if len(x_roi) == 0:
        raise ValueError("The provided x_range contains no data points.")

    # calculate derivative on whole smoothed data, then slice to ROI
    dy = np.gradient(y_smooth, x)
    dy_roi = dy[mask]

    if(object_type == 'bright'):
        # find the rising edge: the steepest positive slope in the ROI
        rising_idx = np.argmax(dy_roi)
        # find end of object (after rising edge)
        dy_roi_after_rising = dy_roi[rising_idx + 1:]
        # find steepest negative slope in smaller slice 
        # must add back (rising_idx + 1) to the result so the index maps correctly
        falling_idx = (rising_idx + 1) + np.argmax(dy_roi_after_rising)
    elif(object_type == 'dark'):
        # find the falling edge: the steepest negative slope in the ROI
        falling_idx = np.argmin(dy_roi)
        # find end of object (after falling edge)
        dy_roi_after_falling = dy_roi[falling_idx + 1:]
        # find steepest positive slope in smaller slice 
        # must add back (rising_idx + 1) to the result so the index maps correctly
        rising_idx = (falling_idx + 1) + np.argmax(dy_roi_after_falling)
    else:
        raise ValueError("Invalid object_type. Must be 'dark' or 'bright'.")
    
    x_start = x_roi[falling_idx]
    x_end = x_roi[rising_idx]
    width = x_end - x_start
    
    #show plot: for validation of calculated edges
    if show_plot:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        
        # plot raw and smoothed data
        ax1.scatter(x, y, color='gray', alpha=0.3, s=10, label='Raw Data')
        ax1.plot(x, y_smooth, color='blue', linewidth=2, label='Smoothed Data')

        # highlight the ROI
        if x_range is not None:
            ax1.axvspan(x_range[0], x_range[1], color='yellow', alpha=0.1, label='Search Range (ROI)')
            
        # draw edges
        ax1.axvline(x_start, color='red', linestyle='--', label=f'Start: {x_start:.1f}')
        ax1.axvline(x_end, color='green', linestyle='--', label=f'End: {x_end:.1f}')
            
        ax1.set_title(f'Cone Width Calculation at Row {row_idx}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Intensity')
        ax1.legend(loc='upper right')

        plt.xlim(750,2750)
        
        plt.tight_layout()
        plt.savefig(f'sh_calib/deriv_fits/cone_row_{row_idx}')
    
    return width, x_start, x_end

def find_left_edge(x, y, row_idx, x_range=None, save_plot=False, object_type = 'dark'):
    """
    Uses first derivative analysis to find the left edge of bright region in a 1D intensity lineout. 
    Alternately, could be used to find right edge of a dark region.
    Plotting optional but recommended for validation
    """
    # smooth entire array for derivative calculation
    y_smooth = savgol_filter(y, window_length=51, polyorder=3)
    
    # create ROI if x_range provided
    mid = 1607
    if x_range is not None:
        mask = (x >= x_range[0]) & (x <= x_range[1])
    else:
        mask = np.ones_like(x, dtype=bool) # Use all data if no range given
        
    x_roi = x[mask]
    y_roi = y_smooth[mask]

    # ensure mask captures data
    if len(x_roi) == 0:
        raise ValueError("The provided x_range contains no data points.")

    # calculate derivative on whole smoothed data, then slice to ROI
    dy = np.gradient(y_smooth, x)
    dy_roi = dy[mask]
    
    # find falling edge: the steepest negative slope in the ROI
    if object_type == 'dark':
        edge_idx = np.argmax(dy_roi)
    else:
        edge_idx = np.argmin(dy_roi)
   
    x_edge = x_roi[edge_idx]

    if save_plot:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        
        ax1.scatter(x, y, color='gray', alpha=0.3, s=10, label='Raw Data')
        ax1.plot(x, y_smooth, color='blue', linewidth=2, label='Smoothed Data')
        
        if x_range is not None:
            ax1.axvspan(x_range[0], x_range[1], color='yellow', alpha=0.1, label='Search Range (ROI)')
            
        ax1.axvline(x_edge, color='red', linestyle='--', label=f'Edge: {x_edge:.1f}')
            
        ax1.set_title(f'Left edge at Row {row_idx}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Intensity')
        ax1.legend(loc='upper right')

        plt.xlim(x_range[0] - 250, x_range[1] + 250)

        plt.tight_layout()
        plt.savefig(f'sh_calib/deriv_fits/right_row_{row_idx}', bbox_inches = 'tight')
        plt.close()
    
    return x_edge

def find_top_edge(x, y, col_idx, y_range=None, save_plot=False, object_type = 'dark'):
    """
    Uses first derivative analysis to find the top edge of bright region in a 1D intensity lineout. 
    Alternately, could be used to find bottom edge of a dark region.
    Plotting optional but recommended for validation
    """
    # smooth data
    x_smooth = savgol_filter(x, window_length=51, polyorder=3)

    # apply mask if given
    if y_range is not None:
        mask = (y >= y_range[0]) & (y <= y_range[1])
    else:
        mask = np.ones_like(x, dtype=bool) # Use all data if no range given
        
    x_roi = x_smooth[mask]
    y_roi = y[mask]

    # mask check if it got data
    if len(x_roi) == 0:
        raise ValueError("The provided x_range contains no data points.")

    # calculate derivative on whole smoothed data, then slice to ROI
    dx = np.gradient(x_smooth, y)
    dx_roi = dx[mask]
    
    # the falling edge is the steepest negative slope in the ROI
    if object_type == 'dark':
        edge_idx = np.argmin(dx_roi)
    else:
        edge_idx = np.argmax(dx_roi)

    y_edge = y_roi[edge_idx]

    if save_plot:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        
        ax1.scatter(x, y, color='gray', alpha=0.3, s=10, label='Raw Data')
        ax1.plot(x_smooth, y, color='blue', linewidth=2, label='Smoothed Data')
        
        if y_range is not None:
            ax1.axvspan(y_range[0], y_range[1], color='yellow', alpha=0.1, label='Search Range (ROI)')
            
        ax1.axvline(y_edge, color='red', linestyle='--', label=f'Edge: {y_edge:.1f}')
            
        ax1.set_title(f'Top edge at Column {col_idx}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Intensity')
        ax1.legend(loc='upper right')

        plt.xlim(y_range[0]-250, y_range[1]+250)

        plt.tight_layout()
        plt.savefig(f'sh_calib/deriv_fits/top_col_{col_idx}')
        plt.close()
        # plt.show()
    
    return y_edge

def find_bottom_edge(x, y, col_idx, y_range=None, save_plot=False):
    """
    Uses first derivative analysis to find the bottom edge of bright region in a 1D intensity lineout.
    Plotting optional but recommended for validation
    """
    # smooth data
    x_smooth = savgol_filter(x, window_length=51, polyorder=3)

    # apply mask if given
    if y_range is not None:
        mask = (y >= y_range[0]) & (y <= y_range[1])
    else:
        mask = np.ones_like(x, dtype=bool) # Use all data if no range given
        
    x_roi = x_smooth[mask]
    y_roi = y[mask]

    # mask check if it got data
    if len(x_roi) == 0:
        raise ValueError("The provided x_range contains no data points.")

    # calculate derivative on whole smoothed data, then slice to ROI
    dx = np.gradient(x_smooth, y)
    dx_roi = dx[mask]
    
    # the falling edge is the steepest negative slope in the ROI
    edge_idx = np.argmax(dx_roi)
   
    y_edge = y_roi[edge_idx]

    if save_plot:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        
        ax1.scatter(x, y, color='gray', alpha=0.3, s=10, label='Raw Data')
        ax1.plot(x_smooth, y, color='blue', linewidth=2, label='Smoothed Data')
        
        if y_range is not None:
            ax1.axvspan(y_range[0], y_range[1], color='yellow', alpha=0.1, label='Search Range (ROI)')
            
        ax1.axvline(y_edge, color='red', linestyle='--', label=f'Edge: {y_edge:.1f}')
            
        ax1.set_title(f'Bottom edge at Column {col_idx}')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Intensity')
        ax1.legend(loc='upper right')

        plt.xlim(y_range[0]-250, y_range[1]+250)

        plt.tight_layout()
        plt.savefig(f'sh_calib/deriv_fits/cone_bottom_col_{col_idx}')
        plt.close()
        # plt.show()
    
    return y_edge

# ==============================================================================
# STREAK ANALYSIS FUNCTIONS
# ==============================================================================
def plot_vlineout_streak(t_ns, ext, image, return_data = False, plot = False, output = f'streak_images/lineouts/linout.jpg'):
    x_min, x_max, y_min, y_max = ext
    height_px, width_px = image.shape
    
    x_axis_ns = np.linspace(x_min, x_max, width_px)
    y_axis_mm = np.linspace(y_max, y_min, height_px) 

    col_idx = np.argmin(np.abs(x_axis_ns - t_ns))

    actual_x_ns = x_axis_ns[col_idx]
    lineout_data = image[:, col_idx]

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(y_axis_mm, lineout_data, color='tab:green', linewidth=1.5)
        
        plt.title(f'Spatial Profile at t ≈ {actual_x_ns:.2f} ns')
        plt.xlabel('Position [mm]')
        plt.ylabel('Intensity')
        plt.xlim(-2, 4)
        plt.grid(True, alpha=0.5)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()

    if return_data:
        return y_axis_mm, lineout_data

def get_lineouts(streak, times, bg, cf, ext):
    '''
    Averages lineouts over 3 pixels to reduce noise. Returns positions and lineouts as arrays.
    '''
    positions = []
    lineouts = []
    for t in times:
        y1, lo1 = plot_vlineout_streak(t_ns = t - cf[0], ext = ext, image =streak, return_data = True, plot = False, output = f'streak_images/lineouts/linout_t={t}ns.jpg')
        y2, lo2 = plot_vlineout_streak(t_ns = t, ext = ext, image =streak, return_data = True, plot = False, output = f'streak_images/lineouts/linout_t={t}ns.jpg')
        y3, lo3 = plot_vlineout_streak(t_ns = t + cf[0], ext = ext, image =streak, return_data = True, plot = False, output = f'streak_images/lineouts/linout_t={t}ns.jpg')
        lineouts.append(np.average([lo1, lo2, lo3], axis = 0))
        positions.append(np.average([y1, y2, y3], axis = 0))
    positions = np.array(positions)
    lineouts = np.array(lineouts, dtype=object)
    if bg.all() != 0:
        print('background subtracting...')
        lineouts = (lineouts-bg)/bg
    return positions, lineouts
def find_shock_positions(x_data, y_data, times, piston_lim, mag_lim):
    plt.plot(x_data, y_data, color = 'brown', label = f'Lineout', alpha = 0.8)
    
    mask_piston = (x_data >= piston_lim[0]) & (x_data <= piston_lim[1])
    x_fit_p, y_fit_p = x_data[mask_piston],y_data[mask_piston]
    piston_max=(x_fit_p[np.argmax(y_fit_p)])
    piston_min=(x_fit_p[np.argmin(y_fit_p)])
    
    mask_mag = (x_data >= mag_lim[0]) & (x_data <= mag_lim[1])
    x_fit, y_fit = x_data[mask_mag],y_data[mask_mag]
    
    magshock_max = x_fit[np.argmax(y_fit)]
    magshock_min = x_fit[np.argmin(y_fit)]
    
    plt.axvline(magshock_max, color = 'seagreen', label = 'MagShock Max')
    plt.axvline(magshock_min, color = 'limegreen', label = 'MagShock Min')
    
    plt.axvline(piston_max, color = 'deepskyblue', label = 'Piston Max')
    plt.axvline(piston_min, color = 'powderblue', label = 'Piston Min')
        
    plt.xlim(0,8)
    plt.xlim(piston_max-1,piston_max+1.5)
    plt.legend(loc = 'upper left')
    plt.ylim(-1,5)
    plt.title(f'Shock Lineout at t = {times}')
    plt.xlabel('Position [mm]')
    plt.ylabel('Intensity')
    plt.savefig(f'streak_images/lineouts/peaks/t={times}.jpg', bbox_inches = 'tight') 
    plt.clf()

    peaks = [piston_min, piston_max, magshock_min, magshock_max]

    return peaks

def find_shock_positions_3rd(x_data, y_data, times, piston_lim, mag_lim, diag_lim):
    plt.plot(x_data, y_data, color = 'brown', label = f'Lineout', alpha = 0.8)
    
    mask_piston = (x_data >= piston_lim[0]) & (x_data <= piston_lim[1])
    x_fit_p, y_fit_p = x_data[mask_piston],y_data[mask_piston]
    piston_max=(x_fit_p[np.argmax(y_fit_p)])
    piston_min=(x_fit_p[np.argmin(y_fit_p)])
    
    mask_mag = (x_data >= mag_lim[0]) & (x_data <= mag_lim[1])
    x_fit, y_fit = x_data[mask_mag],y_data[mask_mag]
    
    magshock_max = x_fit[np.argmax(y_fit)]
    magshock_min = x_fit[np.argmin(y_fit)]

    mask_diag = (x_data >= diag_lim[0]) & (x_data <= diag_lim[1])
    x_fit_d, y_fit_d = x_data[mask_diag],y_data[mask_diag]
    
    diag_max = x_fit_d[np.argmax(y_fit_d)]
    diag_min = x_fit_d[np.argmin(y_fit_d)]
    
    plt.axvline(magshock_max, color = 'seagreen', label = 'MagShock Max')
    plt.axvline(magshock_min, color = 'limegreen', label = 'MagShock Min')
    
    plt.axvline(piston_max, color = 'deepskyblue', label = 'Piston Max')
    plt.axvline(piston_min, color = 'powderblue', label = 'Piston Min')

    plt.axvline(diag_max, color = 'darkorange', label = 'Diag Max')
    plt.axvline(diag_min, color = 'gold', label = 'Diag Min')
        
    plt.xlim(0,8)
    plt.xlim(piston_max-1,piston_max+1.5)
    plt.legend(loc = 'upper left')
    plt.ylim(-1,5)
    plt.title(f'Shock Lineout at t = {times}')
    plt.xlabel('Position [mm]')
    plt.ylabel('Intensity')
    plt.savefig(f'streak_images/lineouts/peaks/t={times}.jpg', bbox_inches = 'tight') 
    plt.clf()

    peaks = [piston_min, piston_max, magshock_min, magshock_max, diag_min, diag_max]

    return peaks

def get_averaged_lineouts(x_common, t, pos, lo, initial_piston):
    pos_shifted = []
    lineouts_shifted = []
    maxs = []
    mins = []
    
    for i in range(len(t)):

        x_data = np.asarray(pos[i], dtype=float)
        y_data = np.asarray(lo[i], dtype=float)

        min,max = 1,3.1
        # if (i < 20):
        #     min, max = 1.5, 2.5
        shift = (t[i] - t[0])*0.5
        min, max = (initial_piston + shift - 0.3), (initial_piston + shift + 0.3)
        mins.append(min)
        maxs.append(max)
        
        mask = (x_data >= min) & (x_data <= max)
    
        x_fit_p, y_fit_p = x_data[mask],y_data[mask]
        piston_max=(x_fit_p[np.argmax(y_fit_p)])
    
        #plot to verify tracking
        plt.scatter(t[i], piston_max)
        plt.xlabel('Time [ns]')
        plt.ylabel('Piston Position [mm]')
    
        x_shifted = x_data - piston_max
        sort_idx = np.argsort(x_shifted)
        x_sorted = x_shifted[sort_idx]
        y_sorted = y_data[sort_idx]
    
        # Interpolate onto common axis
        y_interp = np.interp(x_common, x_sorted, y_sorted)
    
        # append in descending order for R to L integration
        pos_shifted.append(x_sorted)
        lineouts_shifted.append(y_interp)
    plt.plot(t, mins, color = 'gray')
    plt.plot(t, maxs, color = 'gray')
    plt.title(f'Piston Position from t = {t[0]}-{t[i]} ns')
    plt.show()
    return pos_shifted, lineouts_shifted

def plot_shock_density(x, y_raw, y_noshift, piston_lim, mag_lim, time):

    path_length = 0.5 #0.5 cm
    
    # make mask for piston based on min and max positions
    mask_piston = (x >= piston_lim[0]) & (x <= piston_lim[1])
    x_piston, y_piston = x[mask_piston], y_raw[mask_piston]

    piston_min = x_piston[np.argmin(y_piston)]
    piston_max = x_piston[np.argmax(y_piston)]
    
    p0_piston = [
        0, 0.05, -200,  # Negative dip
        -0.3, 0.05,  350   # Positive peak
    ]
    
    popt_piston, _ = curve_fit(double_gaussian, x_piston, y_piston, p0=p0_piston)
    
    # make mask for magshock based on min and max positions
    mask_mag = (x >= mag_lim[0]) & (x <= mag_lim[1])
    x_mag, y_mag = x[mask_mag], y_raw[mask_mag]
    
    magshock_min = x_mag[np.argmin(y_mag)]
    magshock_max = x_mag[np.argmax(y_mag)]
    
    p0_mag = [
        magshock_min, 0.05, -200,  # Negative dip (Note the negative amplitude!)
        magshock_max, 0.05,  200   # Positive peak
    ]
    
    popt_mag, _ = curve_fit(double_gaussian, x_mag, y_mag, p0=p0_mag)
    
    # combine fits and mask to spatial windows - make everywhere else equal to 0
    y_fit_piston = np.where(mask_piston, double_gaussian(x, *popt_piston), 0.0)
    y_fit_mag    = np.where(mask_mag,    double_gaussian(x, *popt_mag),    0.0)
    
    # get total fit by summing
    y_fit_total = y_fit_piston + y_fit_mag

    x_cm = x/10
    
    # integrate twice
    dn_dx = cumulative_trapezoid(y_fit_total, x_cm, initial=0) 
    density_pi = dens_corr * cumulative_trapezoid(dn_dx, x_cm, initial=0)
    density = (1/path_length)*dens_corr * cumulative_trapezoid(dn_dx, x_cm, initial=0)

    mask_raw = (x > piston_min - 0.5) & (x < magshock_max + 0.3)
    y_0 = np.where(mask_raw, y_raw, 0)
    y_0_noshift = np.where(mask_raw, y_noshift, 0)
    x_cm_0 = x/10
    dn_dx_0 = cumulative_trapezoid(y_0, x_cm_0, initial=0) 
    dn_dx_0_noshift = cumulative_trapezoid(y_0_noshift, x_cm_0, initial=0)
    density_pi_0 = dens_corr * cumulative_trapezoid(dn_dx_0, x_cm_0, initial=0)
    density_pi_0_noshift = dens_corr * cumulative_trapezoid(dn_dx_0_noshift, x_cm_0, initial=0)

    # plot lineout and fit
    fig, axs = plt.subplots(figsize=(7, 5))
    plt.plot(x, y_raw, color='orangered', alpha=0.6, label='Raw Lineout')
    plt.plot(x, y_fit_total, color='royalblue', linewidth=2.5, label='Combined Gaussian Fit')
    plt.title(f'Lineout Fit: t={time}$\pm 0.5$ ns')
    plt.xlim(-2, 2)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Position Relative to Piston Shock [mm]')
    plt.ylabel(r'Intensity')
    plt.savefig(f'streak_images/density_profiles/intensity_{time}.jpg', bbox_inches = 'tight')

    # plot first integration
    fig, axs = plt.subplots(figsize=(7, 5))
    plt.plot(x, dn_dx_0, color='orangered', linewidth=2, alpha = 0.5, linestyle = '--', label = 'Raw Data Integration')
    plt.plot(x, dn_dx, color='royalblue', linewidth=2, label = 'Fit Integration')
    plt.title(r'First Integration: $\frac{dn_e}{dx}$'+ f' at t={time}$\pm 0.5$ ns')
    plt.ylabel('dn/dx')
    plt.xlabel('Position Relative to Piston Shock [mm]')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'streak_images/density_profiles/dndx_{time}.jpg', bbox_inches = 'tight')

    # plot second integration: density profile
    fig, axs = plt.subplots(figsize=(7, 5))
    plt.plot(x, density_pi_0, color='orangered', linewidth=2, label = 'Raw Relative Density', alpha = 0.5, linestyle = '--')
    plt.plot(x, density_pi, color='royalblue', linewidth=2, label = 'Fit Relative Density')
    plt.title(f'Relative Electron Density Profile: t={time}$\pm 0.5$ ns')
    plt.grid(True)
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Position Relative to Piston Shock [mm]')
    plt.ylabel(r'Relative Density')
    plt.savefig(f'streak_images/density_profiles/n_{time}.jpg', bbox_inches = 'tight')

    # returns x axis, y fit, raw dndx, dndx from fit, raw density, density from fit
    return x, y_fit_total, dn_dx_0, dn_dx, dn_dx_0_noshift, density_pi_0, density_pi, density_pi_0_noshift

def plot_shock_density_3rd(x, y_raw, y_noshift, piston_lim, mag_lim, diag_lim, time):

    path_length = 0.5 #0.5 cm
    
    # make mask for piston based on min and max positions
    mask_piston = (x >= piston_lim[0]) & (x <= piston_lim[1])
    x_piston, y_piston = x[mask_piston], y_raw[mask_piston]

    piston_min = x_piston[np.argmin(y_piston)]
    piston_max = x_piston[np.argmax(y_piston)]
    
    p0_piston = [
        0, 0.05, -200,  # Negative dip
        -0.3, 0.05,  350   # Positive peak
    ]
    
    popt_piston, _ = curve_fit(double_gaussian, x_piston, y_piston, p0=p0_piston)
    
    # make mask for magshock based on min and max positions
    mask_mag = (x >= mag_lim[0]) & (x <= mag_lim[1])
    x_mag, y_mag = x[mask_mag], y_raw[mask_mag]
    
    magshock_min = x_mag[np.argmin(y_mag)]
    magshock_max = x_mag[np.argmax(y_mag)]
    
    p0_mag = [
        magshock_min, 0.05, -200,  # Negative dip (Note the negative amplitude!)
        magshock_max, 0.05,  200   # Positive peak
    ]
    
    popt_mag, _ = curve_fit(double_gaussian, x_mag, y_mag, p0=p0_mag)

    # make mask for magshock based on min and max positions
    mask_diag = (x >= diag_lim[0]) & (x <= diag_lim[1])
    x_diag, y_diag = x[mask_diag], y_raw[mask_diag]
    
    diag_min = x_diag[np.argmin(y_diag)]
    diag_max = x_diag[np.argmax(y_diag)]
    print('diag min:', diag_min, 'diag max:', diag_max)
    
    p0_diag = [
        diag_min, 0.05, -200,  # Negative dip (Note the negative amplitude!)
        diag_max, 0.05,  200   # Positive peak
    ]
    
    popt_diag, _ = curve_fit(double_gaussian, x_diag, y_diag, p0=p0_diag)
    
    # combine fits and mask to spatial windows - make everywhere else equal to 0
    y_fit_piston = np.where(mask_piston, double_gaussian(x, *popt_piston), 0.0)
    y_fit_mag    = np.where(mask_mag,    double_gaussian(x, *popt_mag),    0.0)
    y_fit_diag   = np.where(mask_diag,   double_gaussian(x, *popt_diag),   0.0)

    # get total fit by summing
    y_fit_total = y_fit_piston + y_fit_mag + y_fit_diag

    x_cm = x/10
    
    # integrate twice
    dn_dx = cumulative_trapezoid(y_fit_total, x_cm, initial=0) 
    density_pi = dens_corr * cumulative_trapezoid(dn_dx, x_cm, initial=0)
    density = (1/path_length)*dens_corr * cumulative_trapezoid(dn_dx, x_cm, initial=0)

    mask_raw = (x > piston_min - 1.5) & (x < magshock_max + 0.3)
    y_0 = np.where(mask_raw, y_raw, 0)
    y_0_noshift = np.where(mask_raw, y_noshift, 0)
    x_cm_0 = x/10
    dn_dx_0 = cumulative_trapezoid(y_0, x_cm_0, initial=0) 
    dn_dx_0_noshift = cumulative_trapezoid(y_0_noshift, x_cm_0, initial=0)
    density_pi_0 = dens_corr * cumulative_trapezoid(dn_dx_0, x_cm_0, initial=0)
    density_pi_0_noshift = dens_corr * cumulative_trapezoid(dn_dx_0_noshift, x_cm_0, initial=0)

    # plot lineout and fit
    fig, axs = plt.subplots(figsize=(7, 5))
    plt.plot(x, y_raw, color='orangered', alpha=0.6, label='Raw Lineout')
    plt.plot(x, y_fit_total, color='royalblue', linewidth=2.5, label='Combined Gaussian Fit')
    plt.title(f'Lineout Fit: t={time}$\pm 0.5$ ns')
    plt.xlim(-2, 2)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Position Relative to Piston Shock [mm]')
    plt.ylabel(r'Intensity')
    plt.savefig(f'streak_images/density_profiles/third_feat/intensity_{time}.jpg', bbox_inches = 'tight')

    # plot first integration
    fig, axs = plt.subplots(figsize=(7, 5))
    plt.plot(x, dn_dx_0, color='orangered', linewidth=2, alpha = 0.5, linestyle = '--', label = 'Raw Data Integration')
    plt.plot(x, dn_dx, color='royalblue', linewidth=2, label = 'Fit Integration')
    plt.title(r'First Integration: $\frac{dn_e}{dx}$'+ f' at t={time}$\pm 0.5$ ns')
    plt.ylabel('dn/dx')
    plt.xlabel('Position Relative to Piston Shock [mm]')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'streak_images/density_profiles/third_feat/dndx_{time}.jpg', bbox_inches = 'tight')

    # plot second integration: density profile
    fig, axs = plt.subplots(figsize=(7, 5))
    plt.plot(x, density_pi_0, color='orangered', linewidth=2, label = 'Raw Relative Density', alpha = 0.5, linestyle = '--')
    plt.plot(x, density_pi, color='royalblue', linewidth=2, label = 'Fit Relative Density')
    plt.title(f'Relative Electron Density Profile: t={time}$\pm 0.5$ ns')
    plt.grid(True)
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.ylim(bottom=-0.1e16)
    plt.xlabel('Position Relative to Piston Shock [mm]')
    plt.ylabel(r'Relative Density')
    plt.savefig(f'streak_images/density_profiles/third_feat/n_{time}.jpg', bbox_inches = 'tight')

    # options: 'comparison' or 'peaks'
    r = 'peaks'
    if r == 'comparison':
        # returns x axis, y fit, raw dndx, dndx from fit, raw density, density from fit
        return x, y_fit_total, dn_dx_0, dn_dx, dn_dx_0_noshift, density_pi_0, density_pi, density_pi_0_noshift
    elif r == 'peaks':
        return diag_min, diag_max
    else:
        raise ValueError("Invalid return option. Must be 'comparison' or 'peaks'.")