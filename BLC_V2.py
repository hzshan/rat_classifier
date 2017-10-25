"""
PART OF THE QUANTITATIVE BEHAVIORAL ANALYSIS AND MODELING (QBAM) PROJECT

Haozhe Shan
Mason Laboratory, University of Chicago
September 2016

=============================================================================
This is the Bayes-guided Linear-nonlinear Classifier (BLC) V2.

It analyzes and visualizes the positions of tracked animals throughout the time series, with the option to sort data into multiple categories and compare positional differences between them.
=============================================================================


===========================
SOME ABBREVIATIONS
===========================
var -- In the variable names, "var" refers to the sequence of variance of coordinates. It is the time series of the variance of X/Y coordinates on a sliding window.
pos -- It refers to the cumulative distribution of the subject's position in the session.
target -- It refers to the behavior to be predicted. Usually it is a rat being a opener.
control -- It refers to the absence of the behavior to be predicted.
stimulus -- It refers to all the sessions combined (i.e. target+control).


"""
import matplotlib.pyplot as plt
import numpy as np
from BLC_util import processed_data, grid_count, rat_select_rescale
import os
import time
import cPickle as pickle

# ===========================
# SET UP
# ===========================
# HEATMAP SETTINGS
arena_diameter = 60
grid_size = 2.5

# SELECTING A SEGMENT OF THE ENTIRE SESSION
select_time_segment = True
var_selection_starts = 180
var_selection_ends = 1000  # in seconds
variance_analysis_window_size = 90  # in seconds
analyzing_which_rat_in_the_video = 0
grid_selection_starts = 180
grid_selection_ends = 600

# ==========================
# Some Calculations
# ==========================
first_frame_count = int(var_selection_starts / time_step_of_tracking)
last_frame_count = int((var_selection_ends - variance_analysis_window_size) / time_step_of_tracking - 1)
variance_window_frame_count = int(variance_analysis_window_size / time_step_of_tracking)
number_of_windows = int(last_frame_count - first_frame_count)

training_data_source = secondary_analysis_target_header
testing_data = main_analysis_target_header


class group_information:
    def __init__(self, variance_target_triggering_average, variance_stimulus_average, variance_differential_kernel,
                 position_TTA, position_SA, position_DK, targetvar_dr, targetpos_dr,
                 controlvar_dr, controlpos_dr):
        self.vt = variance_target_triggering_average
        self.vs = variance_stimulus_average
        self.vd = variance_differential_kernel
        self.pt = position_TTA
        self.ps = position_SA
        self.pd = position_DK
        self.tvt = targetvar_dr[:, 0]
        self.tvs = targetvar_dr[:, 1]
        self.tvd = targetvar_dr[:, 2]
        self.tpt = targetpos_dr[:, 0]
        self.tps = targetpos_dr[:, 1]
        self.tpd = targetpos_dr[:, 2]
        self.cvt = controlvar_dr[:, 0]
        self.cvs = controlvar_dr[:, 1]
        self.cvd = controlvar_dr[:, 2]
        self.cpt = controlpos_dr[:, 0]
        self.cps = controlpos_dr[:, 1]
        self.cpd = controlpos_dr[:, 2]

openers = np.array([298.1, 298.2, 299.1, 299.2, 300.1, 301.1, 302.1, 302.2, 303.2, 304.2,
                    305.1, 306.1, 308.2, 309.2, 310.1, 311.1, 312.1, 313.1])
nonopeners = np.array([300.2, 301.2, 303.1, 304.1, 305.2, 306.2, 307.1, 307.2, 308.1, 309.1,
                       310.2, 311.2, 312.2, 313.2])
# %%
'''
SECTION 1: TRAINING
IN THIS SECTION, THE MODEL READS DATA FROM THE TRAINING SET, CALCULATES LINEAR FILTERS WITH WHICH IT PERFORMS DIMENSION REDUCTION ON THE TRAINING SET. IT RECORDS THE TRAINING SET
IN THE FORM OF ITS REPRESENTATION IN THE FEATURE SPACE.
'''
# ==========================
# Selection Training Set and Set Exclusion Principles
# ==========================



start_training_set_sampling_at = 1  # for multi-session analysis, put in the index of the videos
end_training_set_sampling_at = 160

# ========================
# Cross Validation
# =======================
training_mask = np.ones(end_training_set_sampling_at + 1)
testing_mask = np.zeros(end_training_set_sampling_at + 1)
for i in range(start_training_set_sampling_at, end_training_set_sampling_at + 1):
    if np.random.rand(1)[0] < 0:
        training_mask[i] = 0
        testing_mask[i] = 1


def secondary_exclusion_principle1():
    if 1 == 2:
        return True
    else:
        return False


def secondary_exclusion_principle2():
    if data_loaded_for_analysis.c == 'yes':
        return True
    else:
        return False


binary_random_sampling = True


def criteria1():
    if data_loaded_for_analysis.o == 'yes' or data_loaded_for_analysis.o == 'opener' or\
                    float(data_loaded_for_analysis.c) in openers:
        return True
    else:
        return False


def criteria2():
    if data_loaded_for_analysis.o == 'no' or data_loaded_for_analysis.o == 'nonopener' or \
                    float(data_loaded_for_analysis.c) in nonopeners:

        return True
    else:
        return False

def criteria3():
    if data_loaded_for_analysis.o == 'yes' or data_loaded_for_analysis.o == 'opener':
        return True
    else:
        return False

def criteria4():
    if data_loaded_for_analysis.o == 'no' or data_loaded_for_analysis.o == 'nonopener':

        return True
    else:
        return False


# ========================
# Training Analysis Setup
# =======================
training_sorted_target_count = 0
training_sorted_control_count = 0
video_counter = -1
acceleration_sum_target = np.zeros(end_training_set_sampling_at - start_training_set_sampling_at + 5)
acceleration_sum_control = np.zeros(end_training_set_sampling_at - start_training_set_sampling_at + 5)

# =========================
# Sampling of Training Set
# =========================
processing_start_time = time.time()
print "##########Training Set Sampling and Analysis Starts###########"
for i in range(start_training_set_sampling_at, end_training_set_sampling_at + 1):
    analyze_video_number = i
    picklename = training_data_source + '%s.pkl' % analyze_video_number

    if os.path.isfile(picklename) == False or (i in main_group_exclusion):  # Primary exclusion principles
        print "Session excluded from sampling, per primary exclusion principles."
        continue

    if training_mask[i] == 0:
        print "Session excluded due to cross validation"
        continue

    data_loaded_for_analysis = pickle.load(open(picklename, 'rb'))
    print "Currently analyzing " + picklename

    if secondary_exclusion_principle1() or secondary_exclusion_principle2():  # Secondary exclusion principles
        print "Session excluded from analysis, per secondary exclusion principles."
        continue
    processed_X_coordinates, processed_Y_coordinates, processed_velocity_series, selected_time_series = rat_select_rescale(
        data_loaded_for_analysis, analyzing_which_rat_in_the_video, select_time_segment, var_selection_starts, var_selection_ends)
    cache_var = np.zeros(number_of_windows)
    cache_vel = np.zeros(number_of_windows)
    for t in range(first_frame_count - first_frame_count, last_frame_count - first_frame_count):
        first_frame_index = t
        last_frame_index = variance_window_frame_count + t
        windowed_X = processed_X_coordinates[first_frame_index:last_frame_index]
        windowed_Y = processed_Y_coordinates[first_frame_index:last_frame_index]
        cache_var[t] = np.var(windowed_X) + np.var(windowed_Y)
        windowed_vel = processed_velocity_series[first_frame_index:last_frame_index]
        cache_vel[t] = np.mean(windowed_vel + 0.0000001)

    processed_X_coordinates, processed_Y_coordinates, processed_velocity_series, selected_time_series = rat_select_rescale(
        data_loaded_for_analysis, analyzing_which_rat_in_the_video, select_time_segment, grid_selection_starts, grid_selection_ends)
    grid, gridxedges, gridyedges, center_X_coordinate, center_Y_coordinate = grid_count(
        processed_X_coordinates, processed_Y_coordinates, processed_X_coordinates, processed_Y_coordinates,
        arena_diameter, grid_size, center_correction=True)

    if criteria1():
        if training_sorted_target_count == 0:
            target_var = cache_var
            target_pos = grid
            target_vel = cache_vel
        else:
            target_var = np.column_stack((target_var, cache_var))
            target_pos = np.dstack((target_pos, grid))
            target_vel = np.column_stack((cache_vel, target_vel))
        training_sorted_target_count = training_sorted_target_count + 1
        acceleration_sum_target[i] = np.sum((np.abs(np.ediff1d(np.ediff1d(processed_X_coordinates))),
                                             np.abs(np.ediff1d(np.ediff1d(processed_Y_coordinates)))))

        print "######Training in progress: sorted into target group"

    if criteria2():
        if training_sorted_control_count == 0:
            control_var = cache_var
            control_pos = grid
            control_vel = cache_vel
        else:
            control_var = np.column_stack((control_var, cache_var))
            control_pos = np.dstack((control_pos, grid))
            control_vel = np.column_stack((cache_vel, control_vel))
        training_sorted_control_count = training_sorted_control_count + 1
        acceleration_sum_control[i] = np.sum((np.abs(np.ediff1d(np.ediff1d(processed_X_coordinates))),
                                              np.abs(np.ediff1d(np.ediff1d(processed_Y_coordinates)))))

        print "######Training in progress:  sorted into control group"

target_pos = target_pos[4:20, :, :]  # Region of interest analysis
control_pos = control_pos[4:20, :, :]  # Region of interest analysis
acceleration_sum_target = np.ma.masked_equal(acceleration_sum_target, 0).compressed()
acceleration_sum_control = np.ma.masked_equal(acceleration_sum_control, 0).compressed()
print "#################Training Set Analysis Finished: Time elapsed: %05s sec." \
      " Target group has %s members." \
      " Control group has %s members." % (time.time() - processing_start_time, training_sorted_target_count,
                                          training_sorted_control_count)

# ===========================
# Computing Linear Filters.
# ===========================
target_triggering_var = np.mean(target_var + 0.0000001, axis=1)
control_triggering_var = np.mean(control_var + 0.0000001, axis=1)
stimulus_var = (target_triggering_var + control_triggering_var) / 2
target_vel_mean = np.mean(target_vel + 0.0000001, axis=1)
control_vel_mean = np.mean(control_vel + 0.0000001, axis=1)

target_triggering_pos = np.mean(target_pos, axis=2)
control_triggering_pos = np.mean(control_pos, axis=2)
stimulus_pos = (target_triggering_pos + control_triggering_pos) / 2

control_triggering_var = control_triggering_var + 0.0000001  # eliminate zero terms
control_triggering_pos = control_triggering_pos + 0.0000001  # eliminate zero terms
diff_var = target_triggering_var / control_triggering_var
diff_pos = target_triggering_pos.flatten() / control_triggering_pos.flatten()

# ======================================
# Training Set Dimension Reduction
# ======================================
print "Performing Multiple Dimension Reductions on Training Set Data."
starttime = time.time()
target_var_reduced = np.zeros((training_sorted_target_count, 3))
control_var_reduced = np.zeros((training_sorted_control_count, 3))
target_pos_reduced = target_var_reduced.copy()
control_pos_reduced = control_var_reduced.copy()

trig_var_kernel = target_triggering_var.T
stim_var_kernel = stimulus_var.T
diff_var_kernel = diff_var.T

trig_pos_kernel = target_triggering_pos.flatten()
stim_pos_kernel = stimulus_pos.flatten()
diff_pos_kernel = diff_pos

for i in xrange(training_sorted_target_count):
    target_var_reduced[i, 0] = np.dot(target_var[:, i], trig_var_kernel)
    target_var_reduced[i, 1] = np.dot(target_var[:, i], stim_var_kernel)
    target_var_reduced[i, 2] = np.dot(target_var[:, i], diff_var_kernel)

    target_pos_reduced[i, 0] = np.dot(target_pos[:, :, i].flatten(), trig_pos_kernel)
    target_pos_reduced[i, 1] = np.dot(target_pos[:, :, i].flatten(), stim_pos_kernel)
    target_pos_reduced[i, 2] = np.dot(target_pos[:, :, i].flatten(), diff_pos_kernel)

for i in xrange(training_sorted_control_count):
    control_var_reduced[i, 0] = np.dot(control_var[:, i], trig_var_kernel)
    control_var_reduced[i, 1] = np.dot(control_var[:, i], stim_var_kernel)
    control_var_reduced[i, 2] = np.dot(control_var[:, i], diff_var_kernel)

    control_pos_reduced[i, 0] = np.dot(control_pos[:, :, i].flatten(), trig_pos_kernel)
    control_pos_reduced[i, 1] = np.dot(control_pos[:, :, i].flatten(), stim_pos_kernel)
    control_pos_reduced[i, 2] = np.dot(control_pos[:, :, i].flatten(), diff_pos_kernel)
target_pos_reduced[:, 0][np.argmax(target_pos_reduced[:, 0])] = target_pos_reduced[:, 0].mean()
target_pos_reduced[:, 1][np.argmax(target_pos_reduced[:, 1])] = target_pos_reduced[:, 1].mean()
target_pos_reduced[:, 2][np.argmax(target_pos_reduced[:, 2])] = target_pos_reduced[:, 2].mean()

training_data = group_information(target_triggering_var, stimulus_var, diff_var, target_triggering_pos, stimulus_pos,
                                  diff_pos, target_var_reduced, target_pos_reduced, control_var_reduced,
                                  control_pos_reduced)
print "Training Set Dimension Reduction Complete. Time elapsed %s sec." % (time.time() - starttime)
pickle.dump(training_data, open("BNC_training_data.pkl", 'wb'))

# %%
training_data = pickle.load(open('BNC_training_data.pkl', 'rb'))



# ============================
# Plot Training Set Data
# ============================
plt.figure()
plt.subplot(231)
plt.plot(xrange(int(len(trig_var_kernel))), trig_var_kernel / target_vel_mean, color="red", label='Helpers')
plt.plot(xrange(int(len(control_triggering_var))), control_triggering_var / control_vel_mean, color="black",
         label='Non-helpers')
plt.xlabel('Time (sec)')
plt.ylabel('Mean variance sequence')
plt.title('Mean Variance Between Groups')
plt.legend(loc=4)

plt.subplot(232)
plt.imshow(target_triggering_pos, interpolation='None', cmap="Reds", vmin=0, vmax=np.max(target_triggering_pos))
plt.xlabel('Arena,X-axis')
plt.ylabel('Arena,Y-axis')
plt.title('CPH Map of To-Be-Helpers')

plt.subplot(233)
plt.imshow(control_triggering_pos, interpolation='None', cmap="Reds", vmin=0, vmax=np.max(target_triggering_pos))
plt.xlabel('Arena,X-axis')
plt.ylabel('Arena,Y-axis')
plt.title('CPH of To-Be-Non-Helpers')
#
plt.subplot(234)
plt.scatter(training_data.tvt, training_data.tpt, color="red")
plt.scatter(training_data.cvt, training_data.cpt, color="black")
plt.scatter(np.mean(training_data.tvt), np.mean(training_data.tpt), color="red", s=200)
plt.scatter(np.mean(training_data.cvt), np.mean(training_data.cpt), color="black", s=200)

plt.xlabel('Dimension Reduction, Variance')
plt.ylabel('Dimension Reduction, Position')
plt.title('Clustering Using Target Triggering Average as Kernels')

plt.subplot(235)
plt.scatter(training_data.tvs, training_data.tps, color="red")
plt.scatter(training_data.cvs, training_data.cps, color="black")
plt.scatter(np.mean(training_data.tvs), np.mean(training_data.tps), color="red", s=200)
plt.scatter(np.mean(training_data.cvs), np.mean(training_data.cps), color="black", s=200)

plt.xlabel('Dimension Reduction, Variance')
plt.ylabel('Dimension Reduction, Position')
plt.title('Clustering Using Stimulus Average as Kernels')

plt.subplot(236)
plt.scatter(training_data.tvd, training_data.tpd, color="red")
plt.scatter(training_data.cvd, training_data.cpd, color="black")
plt.scatter(np.mean(training_data.tvd), np.mean(training_data.tpd), color="red", s=200)
plt.scatter(np.mean(training_data.cvd), np.mean(training_data.cpd), color="black", s=200)
plt.xlabel('Dimension Reduction, Variance')
plt.ylabel('Dimension Reduction, Position')
plt.title('Clustering Using Differential Kernels')

'''
SECTION 2: TESTING AND PREDICTION
IN THIS SECTION, THE MODEL READS DATA FROM THE TESTING SET, PERFORMS DIMENSION REDUCTION ON THEM USING LINEAR FILTERS ACQUIRED FROM TRAINING, AND MAKES PREDICTIONS.
'''
# ==========================
# Selection Testing Set and Set Exclusion Principles
# ==========================

start_testing_set_sampling_at = 1  # for multi-session analysis, put in the index of the videos
end_testing_set_sampling_at = 225


def secondary_exclusion_principle1():
    if 1 == 2:
        return True
    else:
        return False


def secondary_exclusion_principle2():
    if 1 == 2:
        return True
    else:
        return False


binary_random_sampling = True

# ========================
# Testing Analysis Setup
# =======================
video_counter = -1
target_count = 0
control_count = 0
testing_starttime = time.time()
# =========================
# Sampling of Testing Set
# =========================
for i in range(start_testing_set_sampling_at, end_testing_set_sampling_at + 1):
    analyze_video_number = i
    picklename = testing_data + '%s.pkl' % analyze_video_number

    if os.path.isfile(picklename) == False or (i in main_group_exclusion):  # Primary exclusion principles
        print "Session excluded from testing, per primary exclusion principles."
        continue

#        print "Session excluded due to cross validation"
#        continue

    data_loaded_for_analysis = pickle.load(open(picklename, 'rb'))
    print "Currently analyzing " + picklename

    if secondary_exclusion_principle1() or secondary_exclusion_principle2():  # Secondary exclusion principles
        print "Session excluded from testing, per secondary exclusion principles."
        continue

    processed_X_coordinates, processed_Y_coordinates, processed_velocity_series, selected_time_series = rat_select_rescale(
        data_loaded_for_analysis, analyzing_which_rat_in_the_video,
        select_time_segment, var_selection_starts, var_selection_ends)

    cache_var = np.zeros(number_of_windows)

    for t in range(first_frame_count - first_frame_count, last_frame_count - first_frame_count):
        first_frame_index = t
        last_frame_index = variance_window_frame_count + t
        windowed_X = processed_X_coordinates[first_frame_index:last_frame_index]
        windowed_Y = processed_Y_coordinates[first_frame_index:last_frame_index]
        cache_var[t] = np.var(windowed_X) + np.var(windowed_Y)

    processed_X_coordinates, processed_Y_coordinates, processed_velocity_series, selected_time_series = rat_select_rescale(
        data_loaded_for_analysis, analyzing_which_rat_in_the_video, select_time_segment, var_selection_starts, var_selection_ends)
    grid, gridxedges, gridyedges, center_X_coordinate, center_Y_coordinate = grid_count(
        processed_X_coordinates, processed_Y_coordinates, processed_X_coordinates, processed_Y_coordinates,
        arena_diameter, grid_size, center_correction=True)

    if criteria3():
        if target_count == 0:
            testing_target_var = cache_var
            testing_target_pos = grid
        else:
            testing_target_var = np.column_stack((testing_target_var, cache_var))
            testing_target_pos = np.dstack((testing_target_pos, grid))
        target_count = target_count + 1
        print "######Testing in progress: sorted into target group"

    if criteria4():
        if control_count == 0:
            testing_control_var = cache_var
            testing_control_pos = grid
        else:
            testing_control_var = np.column_stack((testing_control_var, cache_var))
            testing_control_pos = np.dstack((testing_control_pos, grid))
        control_count = control_count + 1
        print "######Testing in progress: sorted into control group"

testing_target_pos = testing_target_pos[4:20, :, :]  # Region of interest analysis
testing_control_pos = testing_control_pos[4:20, :, :]  # Region of interest analysis

print "#################Testing Set Analysis Finished: Time elapsed: %05s sec. Target group has %s members. Control group has %s members." % (
time.time() - processing_start_time, target_count, control_count)

# ======================================
# Training Set Dimension Reduction
# ======================================
print "Performing Multiple Dimension Reductions on Testing Set Data."
starttime = time.time()
testing_target_var_reduced = np.zeros((target_count, 3))
testing_control_var_reduced = np.zeros((control_count, 3))
testing_target_pos_reduced = testing_target_var_reduced.copy()
testing_control_pos_reduced = testing_control_var_reduced.copy()

for i in xrange(target_count):
    testing_target_var_reduced[i, 0] = np.dot(testing_target_var[:, i], trig_var_kernel)
    testing_target_var_reduced[i, 1] = np.dot(testing_target_var[:, i], stim_var_kernel)
    testing_target_var_reduced[i, 2] = np.dot(testing_target_var[:, i], diff_var_kernel)

    testing_target_pos_reduced[i, 0] = np.dot(testing_target_pos[:, :, i].flatten(), trig_pos_kernel)
    testing_target_pos_reduced[i, 1] = np.dot(testing_target_pos[:, :, i].flatten(), stim_pos_kernel)
    testing_target_pos_reduced[i, 2] = np.dot(testing_target_pos[:, :, i].flatten(), diff_pos_kernel)

for i in xrange(control_count):
    testing_control_var_reduced[i, 0] = np.dot(testing_control_var[:, i], trig_var_kernel)
    testing_control_var_reduced[i, 1] = np.dot(testing_control_var[:, i], stim_var_kernel)
    testing_control_var_reduced[i, 2] = np.dot(testing_control_var[:, i], diff_var_kernel)

    testing_control_pos_reduced[i, 0] = np.dot(testing_control_pos[:, :, i].flatten(), trig_pos_kernel)
    testing_control_pos_reduced[i, 1] = np.dot(testing_control_pos[:, :, i].flatten(), stim_pos_kernel)
    testing_control_pos_reduced[i, 2] = np.dot(testing_control_pos[:, :, i].flatten(), diff_pos_kernel)

testing_data = group_information(target_triggering_var, stimulus_var, diff_var, target_triggering_pos, stimulus_pos,
                                 diff_pos, testing_target_var_reduced, testing_target_pos_reduced,
                                 testing_control_var_reduced, testing_control_pos_reduced)
print "Testing Set Dimension Reduction Complete. Time elapsed %s sec." % (time.time() - starttime)
pickle.dump(testing_data, open("BNC_testing_data.pkl", 'wb'))

# %%
# ============================
# Plot Training Set Data
# ============================

plt.figure()
plt.subplot(2, 3, 1)
plt.plot(xrange(int(len(target_triggering_var))), target_triggering_var, color="red", label='Target')
plt.plot(xrange(int(len(control_triggering_var))), control_triggering_var, color="black", label='Control')
plt.xlabel('Time (sec)')
plt.ylabel('Mean variance sequence')
plt.title('Mean Variance Between Groups')
plt.legend()

plt.subplot(2, 3, 2)
plt.imshow(target_triggering_pos, interpolation='None', cmap="Reds")
plt.xlabel('Arena,X-axis')
plt.ylabel('Arena,Y-axis')
plt.title('Cumulative Position Map of Target Group')

plt.subplot(2, 3, 3)
plt.imshow(control_triggering_pos, interpolation='None', cmap="Reds")
plt.xlabel('Arena,X-axis')
plt.ylabel('Arena,Y-axis')
plt.title('Cumulative Position Map of Control Group')

plt.subplot(2, 3, 4)
plt.scatter(training_data.tvt, training_data.tpt, color="red")
plt.scatter(training_data.cvt, training_data.cpt, color="black")
plt.scatter(testing_data.tvt, testing_data.tpt, color="salmon")
plt.scatter(testing_data.cvt, testing_data.cpt, color="grey")
plt.scatter(np.mean(training_data.tvt), np.mean(training_data.tpt), color="red", s=200)
plt.scatter(np.mean(training_data.cvt), np.mean(training_data.cpt), color="black", s=200)
plt.scatter(np.mean(testing_data.tvt), np.mean(testing_data.tpt), color="salmon", s=200)
plt.scatter(np.mean(testing_data.cvt), np.mean(testing_data.cpt), color="grey", s=200)

plt.xlabel('Dimension Reduction, Variance')
plt.ylabel('Dimension Reduction, Position')
plt.title('Clustering Using Target Triggering Average as Kernels')

plt.subplot(2, 3, 5)
plt.scatter(training_data.tvs, training_data.tps, color="red")
plt.scatter(training_data.cvs, training_data.cps, color="black")
plt.scatter(testing_data.tvs, testing_data.tps, color="salmon")
plt.scatter(testing_data.cvs, testing_data.cps, color="grey")
plt.scatter(np.mean(training_data.tvs), np.mean(training_data.tps), color="red", s=200)
plt.scatter(np.mean(training_data.cvs), np.mean(training_data.cps), color="black", s=200)
plt.scatter(np.mean(testing_data.tvs), np.mean(testing_data.tps), color="salmon", s=200)
plt.scatter(np.mean(testing_data.cvs), np.mean(testing_data.cps), color="grey", s=200)
plt.xlabel('Dimension Reduction, Variance')
plt.ylabel('Dimension Reduction, Position')
plt.title('Clustering Using Stimulus Average as Kernels')

plt.subplot(2, 3, 6)
plt.scatter(training_data.tvd, training_data.tpd, color="red")
plt.scatter(training_data.cvd, training_data.cpd, color="black")
plt.scatter(testing_data.tvd, testing_data.tpd, color="salmon")
plt.scatter(testing_data.cvd, testing_data.cpd, color="grey")
plt.scatter(np.mean(training_data.tvd), np.mean(training_data.tpd), color="red", s=200)
plt.scatter(np.mean(training_data.cvd), np.mean(training_data.cpd), color="black", s=200)
plt.scatter(np.mean(testing_data.tvd), np.mean(testing_data.tpd), color="salmon", s=200)
plt.scatter(np.mean(testing_data.cvd), np.mean(testing_data.cpd), color="grey", s=200)
plt.xlabel('Dimension Reduction, Variance')
plt.ylabel('Dimension Reduction, Position')
plt.title('Clustering Using Differential Kernels')

# %% Finalzing nonlinearity
kernel_selection = 0
grid_count = 100
xt = target_var_reduced[:, kernel_selection] / 1000000000
yt = target_pos_reduced[:, kernel_selection]
xc = control_var_reduced[:, kernel_selection] / 1000000000
yc = control_pos_reduced[:, kernel_selection]
xedges = np.linspace(min(np.append(xt, xc)), max(np.append(xt, xc)), num=grid_count, endpoint=True)
yedges = np.linspace(min(np.append(yt, yc)), max(np.append(yt, yc)), num=grid_count, endpoint=True)
grid1, xedges, yedges = np.histogram2d(xt, yt, bins=(xedges, yedges))
grid2, xedges, yedges = np.histogram2d(xc, yc, bins=(xedges, yedges))
bayesian_normalizing = np.sum(grid1) / (np.sum(grid2) + np.sum(grid1))

prob_grid = np.zeros_like(grid1)
for i in xrange(grid_count - 1):
    for m in xrange(grid_count - 1):
        if grid2[i, m] == 0 and grid1[i, m] == 0:
            prob_grid[i, m] = bayesian_normalizing
        else:
            prob_grid[i, m] = (grid1[i, m] / (grid1[i, m] + grid2[i, m]))

smoothing_para = 10
smooth_grid = scipy.ndimage.gaussian_filter1d(prob_grid, smoothing_para)

smooth_grid = scipy.ndimage.gaussian_filter1d(smooth_grid.T, smoothing_para).T

#
# smooth_grid=smooth_grid*bayesian_normalizing/np.mean(smooth_grid)
plot = plt.figure()
plot1 = plot.add_subplot(111, projection='3d')
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
plot1.plot_surface(xpos, ypos, smooth_grid, rstride=1, cstride=1, cmap=cm.coolwarm,
                   linewidth=0, antialiased=True)
plot1.set_xlabel('Feature:SCV')
plot1.set_ylabel('Feature:CPH')
plot1.set_zlabel('Probability of Being a Helper')

# Automated classification

target_classification_prob = np.zeros_like(testing_target_var_reduced[:, kernel_selection])
for i in xrange(len(target_classification_prob)):
    masking_H, xedges, yedges = np.histogram2d(
        np.repeat(testing_target_var_reduced[i, kernel_selection] / 1000000000, 2),
        np.repeat(testing_target_pos_reduced[i, kernel_selection], 2), bins=(xedges, yedges))
    masking_H = masking_H / 2
    target_classification_prob[i] = np.dot(masking_H.flatten(), smooth_grid.flatten())

target_labels = np.ones_like(target_classification_prob)
empty_target_labels = np.zeros_like(target_labels)
target_labelled_prob = np.column_stack((target_classification_prob, target_labels, empty_target_labels))

control_classification_prob = np.zeros_like(testing_control_var_reduced[:, kernel_selection])
for i in xrange(len(control_classification_prob)):
    masking_H, xedges, yedges = np.histogram2d(
        np.repeat(testing_control_var_reduced[i, kernel_selection] / 1000000000, 2),
        np.repeat(testing_control_pos_reduced[i, kernel_selection], 2), bins=(xedges, yedges))
    masking_H = masking_H / 2
    control_classification_prob[i] = np.dot(masking_H.flatten(), smooth_grid.flatten())
control_labels = np.zeros_like(control_classification_prob)
control_labelled_prob = np.column_stack((control_classification_prob, control_labels, control_labels))

combined_prob = np.row_stack((target_labelled_prob, control_labelled_prob))
array = np.core.records.fromarrays(combined_prob.T, names='Prob,True_label,Classification_label').T
sorted_array = np.sort(array, order='Prob')[::-1]
count = int(len(combined_prob) * bayesian_normalizing)

for i in xrange(count):
    sorted_array['Classification_label'][i] = 1

print sklearn.metrics.normalized_mutual_info_score(sorted_array['True_label'], sorted_array['Classification_label'])
print sklearn.metrics.accuracy_score(sorted_array['True_label'], sorted_array['Classification_label'], normalize=True)
# %%
dummy_MI = np.zeros(100)
dummy_accuracy = np.zeros(100)
dummy_test = sorted_array.copy()
for i in xrange(100):
    dummy_test['Classification_label'] = np.random.binomial(1, bayesian_normalizing, len(dummy_test))

    dummy_MI[i] = sklearn.metrics.adjusted_mutual_info_score(dummy_test['True_label'],
                                                             dummy_test['Classification_label'])
    dummy_accuracy[i] = sklearn.metrics.accuracy_score(dummy_test['True_label'], dummy_test['Classification_label'],
                                                       normalize=True)

print np.mean(dummy_MI)
print np.mean(dummy_accuracy)
