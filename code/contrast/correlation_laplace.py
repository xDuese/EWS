#correlation
import numpy as np
import pandas as pd
import ast
from skimage import io, color, filters, exposure, transform
import scipy.ndimage as ndimage
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def detect_fixations(gaze_data, min_duration=100, max_dispersion=25):
    """
    Detects fixations from raw gaze data using a dispersion-based algorithm.
    """
    fixations = []
    if len(gaze_data) < 2:
        return pd.DataFrame()

    current_fixation_start_index = 0
    current_fixation_end_index = 0
    
    while current_fixation_end_index < len(gaze_data):
        current_gaze_point = gaze_data.iloc[current_fixation_start_index]
        
        while True:
            current_fixation_end_index += 1
            if current_fixation_end_index >= len(gaze_data):
                break
                
            next_gaze_point = gaze_data.iloc[current_fixation_end_index]
            
            dist = np.sqrt((next_gaze_point['gaze_point_x'] - current_gaze_point['gaze_point_x'])**2 +
                           (next_gaze_point['gaze_point_y'] - current_gaze_point['gaze_point_y'])**2)
            
            if dist > max_dispersion:
                break
        
        duration = (gaze_data['device_time_stamp'].iloc[current_fixation_end_index - 1] -
                    gaze_data['device_time_stamp'].iloc[current_fixation_start_index])
        
        if duration >= min_duration:
            fixation_points = gaze_data.iloc[current_fixation_start_index:current_fixation_end_index]
            fixations.append({
                'x': fixation_points['gaze_point_x'].mean(),
                'y': fixation_points['gaze_point_y'].mean(),
                'duration': duration
            })
            
        current_fixation_start_index = current_fixation_end_index
        
    return pd.DataFrame(fixations)


def analyze_correlation_with_kde(image_path, gaze_data_path, kde_bandwidth_scale=1.0):
    """
    Performs image analysis and a more robust gaze data correlation using
    Kernel Density Estimation on detected fixations.
    """
    try:
        img = io.imread(image_path)
        if img.ndim == 2:
            img = color.gray2rgb(img)
        img_height, img_width, _ = img.shape
        print(f"Image '{image_path}' loaded successfully. Dimensions: {img_width}x{img_height}\n")
    except FileNotFoundError:
        print(f"Error: The image file at '{image_path}' was not found. Please provide a valid path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return

    gray_img = color.rgb2gray(img)
    contrast_map = np.abs(filters.laplace(gray_img))
    lab_img = color.rgb2lab(img)
    a_channel = lab_img[:, :, 1]
    b_channel = lab_img[:, :, 2]
    def local_std_filter(arr):
        return np.std(arr)
    std_a = ndimage.generic_filter(a_channel, local_std_filter, size=5)
    std_b = ndimage.generic_filter(b_channel, local_std_filter, size=5)
    color_difference_map = np.sqrt(std_a*2 + std_b*2)

    try:
        df = pd.read_csv(gaze_data_path)
        print(f"Gaze data file '{gaze_data_path}' loaded successfully.")
        print(f"Total rows in raw data: {len(df)}")
    except FileNotFoundError:
        print(f"Error: The gaze data file at '{gaze_data_path}' was not found. Please provide a valid path.")
        return

    # --- Step 1: Calculate Binocular Average ---
    valid_gaze = df[(df['left_gaze_point_validity'] == 1) & (df['right_gaze_point_validity'] == 1)].copy()
    print(f"Rows after binocular validity filter: {len(valid_gaze)}")
    valid_gaze['left_gaze_point_on_display_area'] = valid_gaze['left_gaze_point_on_display_area'].apply(ast.literal_eval)
    valid_gaze['right_gaze_point_on_display_area'] = valid_gaze['right_gaze_point_on_display_area'].apply(ast.literal_eval)
    valid_gaze['gaze_point_x'] = valid_gaze.apply(lambda row: (row['left_gaze_point_on_display_area'][0] + row['right_gaze_point_on_display_area'][0]) / 2 * img_width, axis=1)
    valid_gaze['gaze_point_y'] = valid_gaze.apply(lambda row: (row['left_gaze_point_on_display_area'][1] + row['right_gaze_point_on_display_area'][1]) / 2 * img_height, axis=1)

    # --- Step 2: Detect Fixations ---
    fixations_df = detect_fixations(valid_gaze, min_duration=50, max_dispersion=25)
    print(f"Fixations detected: {len(fixations_df)}")

    if len(fixations_df) < 2:
        print("Not enough valid fixations (less than 2) to perform correlation analysis.")
        return
        
    gaze_points_x = fixations_df['x']
    gaze_points_y = fixations_df['y']

    # --- Step 3: Create KDE Heatmap from Fixations ---
    gaze_points = np.vstack([gaze_points_x, gaze_points_y])
    kde = gaussian_kde(gaze_points, bw_method=kde_bandwidth_scale)
    x_grid, y_grid = np.meshgrid(np.arange(img_width), np.arange(img_height))
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    gaze_heatmap = kde.evaluate(grid_coords).reshape(img_height, img_width)
    
    # --- Step 4: Correlation Analysis ---
    resized_contrast_map = transform.resize(contrast_map, gaze_heatmap.shape, anti_aliasing=True)
    resized_color_diff_map = transform.resize(color_difference_map, gaze_heatmap.shape, anti_aliasing=True)
    
    gaze_heatmap_flat = gaze_heatmap.flatten()
    contrast_map_flat = resized_contrast_map.flatten()
    color_diff_map_flat = resized_color_diff_map.flatten()
    
    r_contrast, p_value_contrast = pearsonr(gaze_heatmap_flat, contrast_map_flat)
    r_color_diff, p_value_color_diff = pearsonr(gaze_heatmap_flat, color_diff_map_flat)

    print("--- Correlation Results ---")
    print(f"Correlation between gaze points and high contrast areas (Laplace):")
    print(f"   Pearson r = {r_contrast:.4f}")
    
    print("Correlation between gaze points and large color differences (LAB Chroma):")
    print(f"   Pearson r = {r_color_diff:.4f}")

# Example usage with adjustable bandwidth and fixation detection
image_path = "/Users/timnotzold/Downloads/id014_ort_.jpg"
gaze_data_path = "/Users/timnotzold/Downloads/Proband0_id014_ort_.csv"
analyze_correlation_with_kde(image_path, gaze_data_path, kde_bandwidth_scale=1.0)