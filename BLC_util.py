import numpy as np

#=================================
#RAW DATA TREATMENT
#The file processor (which is a separate script) can perform treatment on raw data time series (low-pass filtering). Would you like that?
#=================================
use_treated_files=False

#=================================
#EXCLUSION PRINCIPLES
#Are there data files that you do not want to include in the analysis? Put them in here.
#=================================
main_group_exclusion=np.array([999])
secondary_group_exclusion=np.array([5,10])

#=================================
#TRACKING DETAILS
#How many rats are analyzed in each tracking file? What's the time step of tracking? Get these parameters from Ethovision.
#=================================
number_of_rats_per_video=1
time_step_of_tracking=0.033
arena_diameter=60

def loadseries(first_row_for_loading,last_row_for_loading,row_title,spreadsheet_name):
    i=0
    target=np.zeros((last_row_for_loading-first_row_for_loading+1))
    datarange='%s%i:%s%i' % (row_title,first_row_for_loading,row_title,last_row_for_loading)
    loaded_vector_length=last_row_for_loading-first_row_for_loading+1
    for row in spreadsheet_name.iter_rows(datarange):
        for cell in row:
            if cell.value=='-':
                target[i]=0
            else:
                target[i]=cell.value
            i=i+1
    return target,loaded_vector_length;


class processed_data:
    def __init__(self, filename, condition, group, trapped_id, date, ws0timeline,
                 ws0x, ws0y, ws0velocity, opener):
        self.n = filename
        self.c = condition
        self.o = opener
        self.g = group
        self.tid = trapped_id
        self.d = date
        self.t = ws0timeline
        self.x0 = ws0x
        self.y0 = ws0y
        self.v0 = ws0velocity


def grid_count(reference_X_series, reference_Y_series, counted_X_series, counted_Y_series, arena_diameter,
                        grid_size, center_correction):
    x_coordinates = counted_X_series
    y_coordinates = counted_Y_series

    center_X_coordinate = 0
    center_Y_coordinate = 0
    center_Y_grid_coordinate = int(arena_diameter / grid_size / 2)
    center_X_grid_coordinate = int(arena_diameter / grid_size / 2)
    xedges = np.arange(-0.5 * arena_diameter + center_X_coordinate, 0.5 * arena_diameter + center_X_coordinate,
                       grid_size)
    yedges = np.arange(-0.5 * arena_diameter + center_Y_coordinate, 0.5 * arena_diameter + center_Y_coordinate,
                       grid_size)
    H, xedges, yedges = np.histogram2d(x_coordinates, y_coordinates, bins=(xedges, yedges))
    grid = H * time_step_of_tracking
    if center_correction:
        central_region_upperbound = 5 + int(center_Y_grid_coordinate)
        central_region_lowerbound = -5 + int(center_Y_grid_coordinate)
        central_region_leftbound = -5 + int(center_X_grid_coordinate)
        central_region_rightbound = 5 + int(center_X_grid_coordinate)
        central_region = grid[central_region_leftbound:central_region_rightbound,
                         central_region_lowerbound:central_region_upperbound]
        central_region = np.clip(central_region, np.min(central_region), np.percentile(central_region, 80))
        grid[central_region_leftbound:central_region_rightbound,
        central_region_lowerbound:central_region_upperbound] = central_region

    return grid, xedges, yedges, center_X_grid_coordinate, center_Y_grid_coordinate;



def rat_select_rescale(data_loaded_for_analysis,analyzing_which_rat_in_the_video,select_or_not,selection_starts,selection_ends):
    # =================================
    # (ADVANCED)ARENA PERIMETER
    # When importing coordinates and performing calibration, the package needs to know where the edges of the arena are. This is done by
    # analyzing coordinates on the margins of the group. A larger sampling sample size gets a smoother arena perimeter but risks loss of some data points.
    # =================================
    left_edge_smoothing_sample_size = 100
    right_edge_smoothing_sample_size = 100
    upper_edge_smoothing_sample_size = 100
    lower_edge_smoothing_sample_size = 100  # When determining the edges of arena, how many samples of data points to take. Larger number allows the script to remove outliers better, but may omit marginal real data points.

    if analyzing_which_rat_in_the_video==0:
        x_coordinates=data_loaded_for_analysis.x0
        y_coordinates=data_loaded_for_analysis.y0
        velocity_series=data_loaded_for_analysis.v0
    if analyzing_which_rat_in_the_video==1:
        x_coordinates=data_loaded_for_analysis.x1
        y_coordinates=data_loaded_for_analysis.y1
        velocity_series=data_loaded_for_analysis.v1
    x_axis_lowerbound=np.mean(np.sort(x_coordinates,axis=None)[:left_edge_smoothing_sample_size])
    x_axis_upperbound=np.mean(np.sort(x_coordinates,axis=None)[::-1][:right_edge_smoothing_sample_size])
    y_axis_lowerbound=np.mean(np.sort(y_coordinates,axis=None)[:lower_edge_smoothing_sample_size])
    y_axis_upperbound=np.mean(np.sort(y_coordinates,axis=None)[::-1][:upper_edge_smoothing_sample_size])
    for i in xrange(int(len(x_coordinates))):
        if x_coordinates[i]<(x_axis_lowerbound) or x_coordinates[i]>x_axis_upperbound or y_coordinates[i]<y_axis_lowerbound or y_coordinates[i]>y_axis_upperbound:
            x_coordinates[i]=0
            y_coordinates[i]=0
    center_X_coordinate=1/2*(np.max(x_coordinates)+np.min(x_coordinates))
    center_Y_coordinate=1/2*(np.max(y_coordinates)+np.min(y_coordinates))
    xscale=arena_diameter/(np.max(x_coordinates)-np.min(x_coordinates))
    yscale=arena_diameter/(np.max(y_coordinates)-np.min(y_coordinates))
    rescaled_X_coordinates=(x_coordinates-center_X_coordinate)*xscale
    rescaled_Y_coordinates=(y_coordinates-center_Y_coordinate)*yscale
    rescaled_velocity_series=velocity_series
    if select_or_not:
        selection_starts=int(selection_starts/time_step_of_tracking)
        selection_ends=int(selection_ends/time_step_of_tracking)
        rescaled_X_coordinates=rescaled_X_coordinates[selection_starts:selection_ends]
        rescaled_Y_coordinates=rescaled_Y_coordinates[selection_starts:selection_ends]
        rescaled_velocity_series=rescaled_velocity_series[selection_starts:selection_ends]
        selected_time_series=data_loaded_for_analysis.t[selection_starts:selection_ends]
    else:
        rescaled_X_coordinates=rescaled_X_coordinates
        rescaled_Y_coordinates=rescaled_Y_coordinates
        rescaled_velocity_series=rescaled_velocity_series
        selected_time_series=None
    return rescaled_X_coordinates,rescaled_Y_coordinates,rescaled_velocity_series,selected_time_series


