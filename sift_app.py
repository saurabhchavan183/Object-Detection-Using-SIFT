# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from numpy import (all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace,
                   unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32)
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
# Removed matplotlib import as we'll use st.image
from skimage import io, color
import io as python_io # Renamed to avoid conflict with skimage.io

# --- Constants ---
ASSUMED_BLUR = 0.5
SIGMA = 1.6
NUM_INTERVALS = 3
IMAGE_BORDER_WIDTH = 5
CONTRAST_THRESHOLD = 0.04
EIGENVALUE_RATIO = 10
NUM_ATTEMPTS_UNTIL_CONVERGENCE = 5
FLOAT_TOLERANCE = 1e-7
DESCRIPTOR_WINDOW_WIDTH = 4
DESCRIPTOR_NUM_BINS = 8
DESCRIPTOR_SCALE_MULTIPLIER = 3
DESCRIPTOR_MAX_VALUE = 0.2
MIN_MATCH_COUNT_THRESHOLD = 10 # User defined threshold

# --- SIFT Implementation Functions (Copied directly from the notebook) ---

# Note: Removed plotting functions like show_images, plot_images, draw_histogram, draw_keypoints
# as they are not directly needed for the final Streamlit output.
# The core logic remains.

def generateBaseImage(image, sigma, assumed_blur):
    """Upsamples image by 2 and applies Gaussian blur"""
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def computeNumberOfOctaves(image_shape):
    """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)"""
    return int(round(log(min(image_shape)) / log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper."""
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space image pyramid"""
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return array(gaussian_images, dtype=object)

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid"""
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

# --- Scale-space extrema related ---
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=CONTRAST_THRESHOLD):
    """Find pixel positions of all scale-space extrema in the image pyramid"""
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, EIGENVALUE_RATIO, NUM_ATTEMPTS_UNTIL_CONVERGENCE)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise"""
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= first_subimage) and \
                   all(center_pixel_value >= third_subimage) and \
                   all(center_pixel_value >= second_subimage[0, :]) and \
                   all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
                   all(center_pixel_value <= third_subimage) and \
                   all(center_pixel_value <= second_subimage[0, :]) and \
                   all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=EIGENVALUE_RATIO, num_attempts_until_convergence=NUM_ATTEMPTS_UNTIL_CONVERGENCE):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors"""
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        # st.write(f"DEBUG: Extremum outside image at attempt {attempt_index+1}")
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        # st.write("DEBUG: Maximum number of attempts reached for convergence")
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        # Check Hessian determinant and trace condition (for eliminating edge responses)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            # Extremum location: original (j, i) + update * 2^octave_index
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            # Octave encoding: octave + layer * 2^8 + round((update_scale + 0.5) * 255) * 2^16
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            # Size: sigma * 2^(image_index + update_scale / num_intervals) * 2^(octave_index + 1) ( +1 because image was doubled)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    # st.write(f"DEBUG: Failed contrast or Hessian check. Value: {abs(functionValueAtUpdatedExtremum) * num_intervals}, Det: {xy_hessian_det}, Ratio condition: {eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det if xy_hessian_det > 0 else 'N/A'}")
    return None


def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h=1."""
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h=1."""
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint."""
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    # Calculate scale, radius, and weight factor for orientation histogram
    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)

    # Iterate over image region around keypoint to build orientation histogram
    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    # Calculate weight based on Gaussian function and distance from keypoint
                    weight = exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # Smooth the histogram
    smooth_histogram = zeros(num_bins)
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.

    orientation_max = max(smooth_histogram)
    # Find peaks in the smoothed histogram
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]

    # Process peaks to assign orientations
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            # Handle division by zero or very small denominator
            denominator = left_value - 2 * peak_value + right_value
            if abs(denominator) < FLOAT_TOLERANCE:
                 interpolated_peak_index = peak_index # Avoid division by zero, use original peak
            else:
                 interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / denominator) % num_bins

            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < FLOAT_TOLERANCE:
                orientation = 0
            # Create new keypoint with assigned orientation
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)

    # If no peaks are found, return the original keypoint without orientation (or handle as needed)
    if not keypoints_with_orientations:
        # Option: return the keypoint with a default orientation (e.g., 0) or simply skip it
        # For now, let's return an empty list, meaning this keypoint might be discarded later if orientation is crucial
        pass # Or potentially: keypoints_with_orientations.append(keypoint)

    return keypoints_with_orientations # Return list of keypoints with orientations


# --- Duplicate keypoint removal ---
def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2"""
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints"""
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        # Check if the keypoints are effectively the same
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

# --- Keypoint scale conversion ---
def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size"""
    converted_keypoints = []
    for keypoint in keypoints:
        # Adjust point coordinates and size by factor of 0.5
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        # Adjust octave value (subtract 1 from octave part)
        # Bitwise AND with ~255 (11111111 00000000) keeps higher bits
        # Bitwise AND with 255 (00000000 11111111) isolates lower bits (original octave)
        # Then subtract 1 from the original octave and combine
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints


# --- Descriptor generation ---
def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint octave value"""
    octave = keypoint.octave & 255 # Lower 8 bits
    layer = (keypoint.octave >> 8) & 255 # Next 8 bits
    if octave >= 128: # Handle negative octaves (if octave is signed char)
        octave = octave | -128
    # Calculate scale based on octave
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=DESCRIPTOR_WINDOW_WIDTH, num_bins=DESCRIPTOR_NUM_BINS, scale_multiplier=DESCRIPTOR_SCALE_MULTIPLIER, descriptor_max_value=DESCRIPTOR_MAX_VALUE):
    """Generate descriptors for each keypoint"""
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        # Ensure octave index is valid for gaussian_images array (needs +1 adjustment due to base image doubling)
        # Also ensure layer index is valid
        if octave + 1 >= len(gaussian_images) or layer >= len(gaussian_images[octave + 1]):
             #st.warning(f"Skipping keypoint due to invalid octave/layer index: Octave {octave}, Layer {layer}")
             continue # Skip this keypoint if indices are out of bounds

        gaussian_image = gaussian_images[octave + 1][layer] # +1 because octave 0 corresponds to the doubled base image
        num_rows, num_cols = gaussian_image.shape
        # Scale keypoint coordinates to the octave's image scale
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle # Use keypoint orientation
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        # Initialize histogram tensor (add 2 to dimensions for border handling during interpolation)
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))

        # Calculate descriptor window size and half-width
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        # Ensure half_width calculation uses valid keypoint size
        if keypoint.size <= 0:
             #st.warning(f"Skipping keypoint due to non-positive size: {keypoint.size}")
             continue # Skip keypoint if size is invalid

        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5)) # sqrt(2) for diagonal length
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2))) # Clamp to image bounds

        # Collect gradient magnitudes and orientations within the descriptor window
        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                # Rotate coordinates according to keypoint orientation
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle

                # Calculate spatial bin indices (row_bin, col_bin)
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5

                # Check if the point is within the central grid region
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    # Calculate window coordinates and check image bounds
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        # Calculate gradient magnitude and orientation at the sample point
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        # Calculate weight based on Gaussian function
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        # Append data for trilinear interpolation
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree) # Relative orientation

        # Perform trilinear interpolation to populate the histogram tensor
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor

            # Handle orientation bin wrapping
            if orientation_bin_floor < 0: orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins: orientation_bin_floor -= num_bins

            # Calculate interpolation weights
            c1 = magnitude * row_fraction; c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction; c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction; c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction; c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction; c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction; c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction; c000 = c00 * (1 - orientation_fraction)

            # Distribute magnitude to neighboring bins in the histogram tensor (adjust indices by +1 due to padding)
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        # Flatten the central part of the histogram tensor to create the descriptor vector
        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten() # Remove padding

        # Normalize and threshold the descriptor vector
        descriptor_norm = norm(descriptor_vector)
        # Check for zero norm before division
        if descriptor_norm < FLOAT_TOLERANCE:
            #st.warning("Descriptor norm is close to zero, skipping normalization.")
            # Handle this case, e.g., by setting descriptor to zeros or skipping
             descriptor_vector = zeros_like(descriptor_vector) # Or continue
        else:
            threshold = descriptor_norm * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(norm(descriptor_vector), FLOAT_TOLERANCE) # Re-normalize after thresholding

        # Convert to unsigned char representation (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)

    return array(descriptors, dtype='float32')


# --- Template Matching (Modified for Streamlit) ---
def draw_boundary_and_label(img2, dst, label='Object Found'):
    """Draws boundary and label on the image."""
    # Draw the bounding box in the test image in Red color for visibility
    line_color = (0, 0, 255)  # Red color in BGR
    line_thickness = 3
    # Make sure dst is integer type for polylines
    img2_bound = cv2.polylines(img2.copy(), [np.int32(dst)], True, line_color, line_thickness, cv2.LINE_AA)

    # Put the text label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0 # Made font larger
    font_color = (0, 0, 255)  # Red color in BGR
    font_thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_x = 10
    text_y = text_size[1] + 10
    cv2.putText(img2_bound, label, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return img2_bound

def template_matching(img1_gray, img2_gray, img1_color, img2_color, keypoints1, keypoints2, descriptors1, descriptors2, min_match_count):
    """Performs FLANN based matching and returns results."""
    kp1 = keypoints1
    kp2 = keypoints2
    des1 = descriptors1
    des2 = descriptors2

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        st.warning("Not enough descriptors to perform matching.")
        return None, 0 # Return None for image, 0 for matches

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1 # Use 1 for OpenCV >= 3
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        st.error(f"Error during FLANN matching: {e}")
        st.error("This might happen if descriptors have incompatible types or dimensions.")
        st.error(f"Desc1 shape: {des1.shape}, dtype: {des1.dtype}")
        st.error(f"Desc2 shape: {des2.shape}, dtype: {des2.dtype}")
        return None, 0

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    num_good_matches = len(good_matches)

    # --- Visualization (only if enough matches found) ---
    if num_good_matches > min_match_count:
        # Estimate homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                st.warning("Homography calculation failed.")
                # Optionally still draw matches without bounding box
                img2_display = img2_color.copy() # Use original color image 2 for display
            else:
                # Draw bounding box on image 2
                h, w = img1_gray.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                # Draw boundary on a copy of image 2 color
                img2_display = draw_boundary_and_label(img2_color.copy(), dst, label='Object Found')

        except cv2.error as e:
             st.warning(f"Error finding homography or perspective transform: {e}. Will draw matches without bounding box.")
             M = None
             img2_display = img2_color.copy() # Use original color image 2

        # --- Create the combined image with matches drawn ---
        h1, w1 = img1_color.shape[:2]
        h2, w2 = img2_display.shape[:2] # Use shape of potentially modified img2
        nHeight = max(h1, h2)
        nWidth = w1 + w2

        # Create new composite image
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[:h1, :w1] = img1_color # Place image 1 color

        # Adjust placement if heights differ
        h_offset1 = (nHeight - h1) // 2
        h_offset2 = (nHeight - h2) // 2

        newimg[h_offset1:h_offset1+h1, :w1] = img1_color
        newimg[h_offset2:h_offset2+h2, w1:w1+w2] = img2_display # Place potentially modified image 2 color

        # Draw lines for good matches
        line_color = (0, 255, 0) # Green lines
        for m in good_matches:
             # Get points from original keypoints
             pt1_coords = kp1[m.queryIdx].pt
             pt2_coords = kp2[m.trainIdx].pt

             # Adjust coordinates for the composite image
             pt1 = (int(pt1_coords[0]), int(pt1_coords[1] + h_offset1))
             pt2 = (int(pt2_coords[0] + w1), int(pt2_coords[1] + h_offset2))

             cv2.line(newimg, pt1, pt2, line_color, 1)

        return newimg, num_good_matches # Return the composite image and match count

    else:
        # Not enough matches found
        return None, num_good_matches # Return None for image, and the low match count


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("SIFT Object Detection from Scratch")

st.markdown("""
Upload two images:
1.  **Image 1 (Query/Template):** Contains the object you want to find.
2.  **Image 2 (Scene):** The image where you want to search for the object.
The app will use a custom SIFT implementation to find keypoints, match them,
and determine if the object is present based on the number of matches found.
""")

st.sidebar.header("Settings")
min_match_input = st.sidebar.number_input("Minimum Match Count Threshold", min_value=1, max_value=100, value=MIN_MATCH_COUNT_THRESHOLD, step=1)

col1, col2 = st.columns(2)

with col1:
    st.header("Image 1 (Query/Template)")
    uploaded_file1 = st.file_uploader("Upload Image 1", type=['jpg', 'png', 'jpeg'], key="uploader1")
    if uploaded_file1 is not None:
        # To read image using skimage.io
        image_bytes1 = uploaded_file1.getvalue()
        img1_color_sk = io.imread(python_io.BytesIO(image_bytes1))
        # Ensure it's RGB for display
        if len(img1_color_sk.shape) == 2: # Grayscale
             img1_color_st = cv2.cvtColor(img1_color_sk, cv2.COLOR_GRAY2RGB)
        elif img1_color_sk.shape[2] == 4: # RGBA
             img1_color_st = cv2.cvtColor(img1_color_sk, cv2.COLOR_RGBA2RGB)
        else:
             img1_color_st = img1_color_sk # Assume RGB
        st.image(img1_color_st, caption='Uploaded Image 1', use_container_width=True)

        # Also read for OpenCV (grayscale)
        file_bytes1 = np.asarray(bytearray(image_bytes1), dtype=np.uint8)
        img1_gray_cv = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)
        img1_gray_cv = img1_gray_cv.astype('float32')
        # Read color for cv2 processing if needed
        img1_color_cv = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)


with col2:
    st.header("Image 2 (Scene)")
    uploaded_file2 = st.file_uploader("Upload Image 2", type=['jpg', 'png', 'jpeg'], key="uploader2")
    if uploaded_file2 is not None:
        # To read image using skimage.io
        image_bytes2 = uploaded_file2.getvalue()
        img2_color_sk = io.imread(python_io.BytesIO(image_bytes2))
        # Ensure it's RGB for display
        if len(img2_color_sk.shape) == 2: # Grayscale
             img2_color_st = cv2.cvtColor(img2_color_sk, cv2.COLOR_GRAY2RGB)
        elif img2_color_sk.shape[2] == 4: # RGBA
             img2_color_st = cv2.cvtColor(img2_color_sk, cv2.COLOR_RGBA2RGB)
        else:
             img2_color_st = img2_color_sk # Assume RGB
        st.image(img2_color_st, caption='Uploaded Image 2', use_container_width=True)

        # Also read for OpenCV (grayscale)
        file_bytes2 = np.asarray(bytearray(image_bytes2), dtype=np.uint8)
        img2_gray_cv = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
        img2_gray_cv = img2_gray_cv.astype('float32')
        # Read color for cv2 processing
        img2_color_cv = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

st.divider()

if st.button("Run SIFT Analysis", disabled=(uploaded_file1 is None or uploaded_file2 is None)):
    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.info("Starting SIFT Analysis... This may take a moment.")
        progress_bar = st.progress(0, text="Initializing...")

        try:
            # --- SIFT Pipeline ---
            # 1. Generate Base Images
            progress_bar.progress(5, text="Generating base images...")
            base_image1 = generateBaseImage(img1_gray_cv, SIGMA, ASSUMED_BLUR)
            base_image2 = generateBaseImage(img2_gray_cv, SIGMA, ASSUMED_BLUR)

            # 2. Compute Octaves
            progress_bar.progress(10, text="Computing number of octaves...")
            num_octaves1 = computeNumberOfOctaves(base_image1.shape)
            num_octaves2 = computeNumberOfOctaves(base_image2.shape)
            if num_octaves1 <= 0 or num_octaves2 <= 0:
                 st.error("Image dimensions are too small to compute octaves. Please use larger images.")
                 st.stop()


            # 3. Generate Gaussian Kernels
            progress_bar.progress(15, text="Generating Gaussian kernels...")
            gaussian_kernels = generateGaussianKernels(SIGMA, NUM_INTERVALS)

            # 4. Generate Gaussian Images (Pyramid)
            progress_bar.progress(25, text="Generating Gaussian image pyramids...")
            gaussian_images1 = generateGaussianImages(base_image1, num_octaves1, gaussian_kernels)
            gaussian_images2 = generateGaussianImages(base_image2, num_octaves2, gaussian_kernels)

            # 5. Generate DoG Images (Pyramid)
            progress_bar.progress(35, text="Generating DoG image pyramids...")
            dog_images1 = generateDoGImages(gaussian_images1)
            dog_images2 = generateDoGImages(gaussian_images2)

            # 6. Find Scale-Space Extrema (Keypoints Pre-Refinement)
            progress_bar.progress(50, text="Finding scale-space extrema...")
            keypoints1_raw = findScaleSpaceExtrema(gaussian_images1, dog_images1, NUM_INTERVALS, SIGMA, IMAGE_BORDER_WIDTH)
            keypoints2_raw = findScaleSpaceExtrema(gaussian_images2, dog_images2, NUM_INTERVALS, SIGMA, IMAGE_BORDER_WIDTH)

            st.write(f"Raw keypoints found: Image 1 = {len(keypoints1_raw)}, Image 2 = {len(keypoints2_raw)}")
            if not keypoints1_raw or not keypoints2_raw:
                st.warning("No raw keypoints found in one or both images. Cannot proceed.")
                st.stop()


            # 7. Remove Duplicate Keypoints
            progress_bar.progress(60, text="Removing duplicate keypoints...")
            keypoints1_unique = removeDuplicateKeypoints(keypoints1_raw)
            keypoints2_unique = removeDuplicateKeypoints(keypoints2_raw)

            st.write(f"Unique keypoints: Image 1 = {len(keypoints1_unique)}, Image 2 = {len(keypoints2_unique)}")
            if not keypoints1_unique or not keypoints2_unique:
                st.warning("No unique keypoints remaining after filtering. Cannot proceed.")
                st.stop()

            # 8. Convert Keypoints to Input Image Size
            progress_bar.progress(65, text="Converting keypoints to input size...")
            keypoints1 = convertKeypointsToInputImageSize(keypoints1_unique)
            keypoints2 = convertKeypointsToInputImageSize(keypoints2_unique)

            # 9. Generate Descriptors
            progress_bar.progress(80, text="Generating descriptors...")
            # Ensure gaussian_images are passed correctly (need original pyramids)
            descriptors1 = generateDescriptors(keypoints1, gaussian_images1)
            descriptors2 = generateDescriptors(keypoints2, gaussian_images2)

            st.write(f"Descriptors generated: Image 1 = {len(descriptors1)}, Image 2 = {len(descriptors2)}")
            if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
                 st.warning("No descriptors were generated for one or both images. Cannot proceed with matching.")
                 st.stop()


            # 10. Template Matching
            progress_bar.progress(90, text="Matching descriptors and generating results...")
            result_image, num_matches = template_matching(img1_gray_cv, img2_gray_cv, img1_color_cv, img2_color_cv, keypoints1, keypoints2, descriptors1, descriptors2, min_match_input)

            progress_bar.progress(100, text="Analysis Complete.")

            # --- Display Results ---
            st.header("Analysis Results")
            st.write(f"Number of Good Matches Found: {num_matches}")
            st.write(f"Required Minimum Matches: {min_match_input}")

            if result_image is not None and num_matches > min_match_input:
                st.success("Object FOUND in Image 2!")
                st.image(result_image, caption=f"SIFT Matches ({num_matches} found) and Bounding Box", use_container_width=True, channels="BGR") # OpenCV uses BGR
            elif num_matches > 0:
                 st.warning(f"Object NOT definitively found (Matches found: {num_matches}, Threshold: {min_match_input})")
                 # Optionally display the matches even if below threshold
                 # if result_image is not None:
                 #    st.image(result_image, caption=f"SIFT Matches ({num_matches} found) - Below Threshold", use_column_width=True, channels="BGR")
                 # else:
                 #    st.write("Could not generate result image (likely due to homography failure even with some matches).")
            else: # num_matches is 0 or result_image is None
                st.error("Object NOT Found (Not enough matches found).")
                st.write("Displaying original images side-by-side:")
                # Display original images side-by-side if no matches found
                st.image([img1_color_st, img2_color_st], caption=['Image 1', 'Image 2'], width=350)


        except Exception as e:
            st.error(f"An error occurred during SIFT analysis: {e}")
            import traceback
            st.error("Traceback:")
            st.code(traceback.format_exc())
        finally:
            # Ensure progress bar completes or hides
             progress_bar.empty()


    else:
        st.warning("Please upload both images before running the analysis.")