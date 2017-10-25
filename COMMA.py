#%%
"""
PART OF THE QUANTITATIVE BEHAVIORAL ANALYSIS AND MODELING (QBAM) PROJECT

Haozhe Shan
Mason Laboratory, University of Chicago
March 2017

=============================================================================
This is the COMMON ANALYTICS ENVIRONMENT (COMMA) V4.

It defines a series of functions and objects, as well as fundamental parameters, across scripts and iterations.
Dependencies: sklearn, scipy, numpy, matplotlib, openpyxl. 
Tested on Python 2.7.
=============================================================================

!!!ALWAYS MAKE SURE THESE PARAMETERS ARE UP-TO-DATE!!!

"""
#=================================
#BASIC MATHEMATICAL OPERATIONS
#=================================
from __future__ import division
import numpy as np
from numpy import pi
from scipy import stats
#=================================
#PATHFINDING
#What's the name of the Excel files that you are using?
#=================================


class sampled_data:
    def __init__(
            self, window_length, end_grouped_analysis_at, selection_starts, selection_ends,
            select_time_segment=True, time_step_of_tracking=0.066, start_grouped_analysis_at=1,
            analyzing_which_rat_in_the_video=0):

        self.analyzing_which_rat_in_the_video = analyzing_which_rat_in_the_video
        self.window_length = window_length
        self.time_step_of_tracking = time_step_of_tracking
        self.start_grouped_analysis_at = start_grouped_analysis_at
        self.end_grouped_analysis_at = end_grouped_analysis_at

        self.selection_starts = selection_starts
        self.selection_ends = selection_ends
        self.select_time_segment = select_time_segment

        self.vector_count_group1 = 0
        self.session_count_g1 = 0
        self.vector_count_group2 = 0
        self.session_count_g2 = 0

        self.processing_start_time = time.time()

        self.window_frame_count = np.rint(self.window_length / self.time_step_of_tracking).astype(int)

        self.session_durations_g1 = np.zeros(self.end_grouped_analysis_at - self.start_grouped_analysis_at + 2)
        self.session_durations_g2 = np.zeros(self.end_grouped_analysis_at - self.start_grouped_analysis_at + 2)
        self.session_durations = np.zeros_like(self.session_durations_g1)
        self.vectors_group1 = np.zeros((int(1000000 / window_length), self.window_frame_count * 2))
        self.vectors_group2 = np.zeros((int(1000000 / window_length), self.window_frame_count * 2))
        self.locations_g1 = np.zeros((1000000, 2))
        self.locations_g2 = self.locations_g1

    def set_principles(self, second_exclusion1, second_exclusion2, criterion1, criterion2):

        self.second_exclusion1 = second_exclusion1
        self.second_exclusion2 = second_exclusion2
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def sample(self, sampling_probability, sampling_redundancy, rotation=True):

        self.sampling_probability = sampling_probability
        self.sampling_probability = sampling_probability

        processing_start_time = time.time()
        for c in range(self.start_grouped_analysis_at, self.end_grouped_analysis_at + 1):
            if np.random.rand() > sampling_probability:
                print "Session skipped in random sampling"
                continue
            picklename = main_analysis_target_header + '%s.pkl' % c

            if os.path.isfile(picklename) == False:
                raise ValueError("File not found.")
                break

            self.data_loaded_for_analysis = pickle.load(open(picklename, 'rb'))  # Load pickle file for analysis
            print "Currently analyzing " + picklename

            processed_X_coordinates, processed_Y_coordinates, processed_velocity_series, selected_time_series = rat_select_rescale(
                self.data_loaded_for_analysis, self.analyzing_which_rat_in_the_video,
                self.select_time_segment, self.selection_starts, self.selection_ends)

            # ================
            # Secondary exclusion: Exclude data from analysis based on the exclusion principles.
            # Typical principles are those that exclude videos with openings or videos from a certain condition.
            # ================
            if eval(self.second_exclusion1) or eval(self.second_exclusion2):  # Secondary exclusion principles
                print "Session excluded from testing, per secondary exclusion principles."
                continue

            position = 0
            for i in xrange(len(processed_X_coordinates) - self.window_frame_count):
                if position > len(processed_X_coordinates) - self.window_frame_count:
                    break
                origin_position = np.matrix((
                    processed_X_coordinates[position], processed_Y_coordinates[position]))

                self.vector_series = np.array((
                    processed_X_coordinates[position:position + self.window_frame_count],
                    processed_Y_coordinates[position:position + self.window_frame_count]))
                position = position + int(self.window_frame_count * (1 - sampling_redundancy))
                position_adjusted_series = self.vector_series - np.tile(origin_position.T, (1, self.window_frame_count))
                direction_basis_vector = position_adjusted_series[:, 1]

                third_edge = np.sqrt(
                    direction_basis_vector[0, 0] * direction_basis_vector[0, 0] + direction_basis_vector[1, 0] *
                    direction_basis_vector[1, 0])

                if third_edge == 0:
                    continue

                cos = direction_basis_vector[0, 0] / third_edge
                sin = direction_basis_vector[1, 0] / third_edge
                rotation_matrix = np.matrix([[cos, -sin], [-sin, -cos]])
                rotated_series = np.zeros_like(position_adjusted_series)

                for m in xrange(len(position_adjusted_series.T)):
                    rotated_series[:, m] = np.matmul(position_adjusted_series[:, m].T, rotation_matrix).T
                self.session_durations[c] += 1

                if rotation == False:
                    rotated_series = position_adjusted_series

                if eval(self.criterion1):
                    self.session_durations_g1[c] = self.session_durations_g1[c] + 1
                    self.vectors_group1[self.vector_count_group1, :] = np.hstack(
                        (rotated_series[0, :], rotated_series[1, :]))
                    self.locations_g1[self.vector_count_group1, :] = np.mean(self.vector_series, axis=1)
                    self.vector_count_group1 += 1

                if eval(self.criterion2):
                    self.session_durations_g2[c] = self.session_durations_g2[c] + 1
                    self.vectors_group2[self.vector_count_group2, :] = np.hstack(
                        (rotated_series[0, :], rotated_series[1, :]))
                    self.locations_g2[self.vector_count_group2, :] = np.mean(self.vector_series, axis=1)

                    self.vector_count_group2 += 1

            print "Action series have been generated and added to the AS library."

        self.g1_session_durations = np.ma.masked_where(self.session_durations_g1 == 0,
                                                       self.session_durations_g1).compressed()
        self.g2_session_durations = np.ma.masked_where(self.session_durations_g2 == 0,
                                                       self.session_durations_g2).compressed()

        self.behavior_vectors_group1 = self.vectors_group1[0:self.vector_count_group1, :]
        self.behavior_vectors_group2 = self.vectors_group2[0:self.vector_count_group2, :]
        self.locations_g1 = self.locations_g1[0:self.vector_count_group1, :]
        self.locations_g2 = self.locations_g2[0:self.vector_count_group2, :]

        print "#################Vectorization: Time elapsed: %05s sec." % (time.time() - processing_start_time)
        return self.session_durations_g1, self.vectors_group1, self.locations_g1, self.session_durations_g2, self.vectors_group2, self.locations_g2

    def PCA(self, n_components):

        """
        Performs principle component analysis on the coordinate trajectories.

        Argument:
        n_components: number of components in the PCA.

        Return:
        self.vector_PCA PCA object
        self.DR_vectors_group1 dimensionality-reduced coordinate trajectories
        self.DR_vectors_group2 dimensionality-reduced coordinate trajectories
        """

        self.joint_AS_library = np.vstack((self.behavior_vectors_group1, self.behavior_vectors_group2))

        self.vector_PCA = PCA(n_components=n_components)
        self.vector_PCA.fit(self.joint_AS_library)
        self.DR_vectors_group1 = self.vector_PCA.transform(self.behavior_vectors_group1)
        self.DR_vectors_group2 = self.vector_PCA.transform(self.behavior_vectors_group2)
        return self.vector_PCA, self.DR_vectors_group1, self.DR_vectors_group2

    def PCA_summary(self):
        print(self.vector_PCA.explained_variance_ratio_)
        print "The first five principles components explain %02s of the variance." % np.sum(
            self.vector_PCA.explained_variance_ratio_[0:5])

        fig = plt.figure()

        self.PC1_x = self.vector_PCA.components_[0, 0:self.window_frame_count]
        self.PC1_y = self.vector_PCA.components_[0, self.window_frame_count:(2 * self.window_frame_count)]
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(self.PC1_x, self.PC1_y, np.linspace(0, self.window_frame_count / 15, self.window_frame_count))
        ax1.set_xlabel('X coordinates')
        ax1.set_ylabel('Y coordinates')
        ax1.set_zlabel('Time (sec)')
        plt.title('principle component 1')

        self.PC2_x = self.vector_PCA.components_[1, 0:self.window_frame_count]
        self.PC2_y = self.vector_PCA.components_[1, self.window_frame_count:(2 * self.window_frame_count)]
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.plot(self.PC2_x, self.PC2_y, np.linspace(0, self.window_frame_count / 15, self.window_frame_count))
        plt.title('principle component 2')

        self.PC3_x = self.vector_PCA.components_[2, 0:self.window_frame_count]
        self.PC3_y = self.vector_PCA.components_[2, self.window_frame_count:(2 * self.window_frame_count)]
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.plot(self.PC3_x, self.PC3_y, np.linspace(0, self.window_frame_count / 15, self.window_frame_count))
        plt.title('principle component 3')

        self.PC4_x = self.vector_PCA.components_[3, 0:self.window_frame_count]
        self.PC4_y = self.vector_PCA.components_[3, self.window_frame_count:(2 * self.window_frame_count)]

        ax4 = fig.add_subplot(234, projection='3d')
        ax4.plot(self.PC4_x, self.PC4_y, np.linspace(0, self.window_frame_count / 15, self.window_frame_count))
        plt.title('principle component 4')

        self.PC5_x = self.vector_PCA.components_[4, 0:self.window_frame_count]
        self.PC5_y = self.vector_PCA.components_[4, self.window_frame_count:(2 * self.window_frame_count)]

        ax5 = fig.add_subplot(235, projection='3d')
        ax5.plot(self.PC5_x, self.PC5_y, np.linspace(0, self.window_frame_count / 15, self.window_frame_count))
        plt.title('principle component 5')


main_analysis_target_header="Raw data-FMT_Version2-Trial   "
secondary_analysis_target_header="Raw data-StrangerHbaitFirst5DaysForHZS-Trial   "


'''
Everything below does not need your input. It just prepares the dependencies for other scripts.
'''

#=================================
#BASIC FUNCTIONAL COMPONENTS
#=================================
from openpyxl import load_workbook, Workbook
import cPickle as pickle
import time
import sys
from sklearn.decomposition import PCA 

#=================================
#VISUALIZATION
#=================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tabulate import tabulate
from matplotlib import gridspec
#=================================
#SIGNAL PROCESSING
#=================================
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import fft
from scipy.signal import butter, lfilter,freqz
import scipy
from sklearn import metrics

from sklearn.preprocessing import normalize
from sklearn import svm
import sklearn
from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors

'''
FUNCTION DEFINITIONS
'''
#============================
#BASIC ANALYTICS INFRASTRUCTURE
#============================




#=========================
#INFORMATION THEORY
#=========================


__all__=['entropy', 'mutual_information', 'entropy_gaussian']
EPS = np.finfo(float).eps

def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor

def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi

    



