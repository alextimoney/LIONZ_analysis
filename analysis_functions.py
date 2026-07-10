import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ==============================================================================
# GENERAL FITTING FUNCTIONS
# ==============================================================================

def gaussian(x, amplitude, mean, std_dev, offset):
    '''
    gaussian fit
    '''
    return -amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) + offset

# ==============================================================================
# CALIBRATION FUNCTIONS
# ==============================================================================

def find_raw_min(x_data, y_data, start_index, end_index):
    '''
    Finds the minimum brightness value in a given range of x_data
    '''
    mask = (x_data >= start_index) & (x_data <= end_index)
    x_fit = x_data[mask]
    y_fit = y_data[mask]

    guess_amp = np.min(y_fit)
    guess_center = x_fit[np.argmin(y_fit)]
    guess_sigma = 10.0 
    guess_offset = 0

    initial_guess = [guess_amp, guess_center, guess_sigma, guess_offset]
    popt, cov = curve_fit(gaussian, x_fit, y_fit, p0=initial_guess)

    return popt, cov, x_fit, y_fit

def get_min_brightness(x_data, y_data, start_index, end_index, plot = True):
    '''
    Fits gaussian to the data to find the minimum brightness and its position
    '''
    popt, cov, x_fit, y_fit = find_raw_min(x_data, y_data, start_index, end_index)
    x_detailed = np.linspace(x_fit.min(), x_fit.max(), 500)

    if plot == True:
        plt.plot(x_detailed, gaussian(x_detailed, *popt), linewidth=2, label='Gaussian Fit 1575-1640', color = cm.batlowS(0))
    
    min_brightness = np.min(gaussian(x_detailed, *popt))
    min_position = x_detailed[np.argmin(gaussian(x_detailed, *popt))]

    return min_brightness, min_position


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

def find_left_edge(x, y, row_idx, x_range=None, save_plot=False):
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
    edge_idx = np.argmax(dy_roi)
   
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
        plt.savefig(f'sh_calib/deriv_fits/left_row_{row_idx}', bbox_inches = 'tight')
        plt.close()
    
    return x_edge

def find_top_edge(x, y, col_idx, y_range=None, save_plot=False):
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
    edge_idx = np.argmin(dx_roi)
   
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

# ==============================================================================
# STREAK ANALYSIS FUNCTIONS
# ==============================================================================
def plot_vlineout_streak(t_ns, ext, image, output = f'streak_images/lineouts/linout.jpg'):
    x_min, x_max, y_min, y_max = ext
    height_px, width_px = image.shape
    
    x_axis_ns = np.linspace(x_min, x_max, width_px)
    y_axis_mm = np.linspace(y_max, y_min, height_px) 

    col_idx = np.argmin(np.abs(x_axis_ns - t_ns))

    actual_x_ns = x_axis_ns[col_idx]
    lineout_data = image[:, col_idx]
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_axis_mm, lineout_data, color='tab:green', linewidth=1.5)
    
    plt.title(f'Spatial Profile at t ≈ {actual_x_ns:.2f} ns')
    plt.xlabel('Position [mm]')
    plt.ylabel('Intensity')
    plt.xlim(-2, 4)
    plt.grid(True, alpha=0.5)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()