#This python script contains functions required for CQA_DLS_GUI to work.
#Ideally it is not required to fix anything in this file (if the setup is the same as proposed by Van Acht et al.)

# DISCLAIMER:
#
# This code is provided "as-is".
# The author is not responsible for any errors, damages, or consequences that arise from the use of
# this code. It is the user's responsibility to thoroughly validate and test the code before using it
# in any medical or clinical environment. Ensure that all necessary precautions are taken and that
# the code complies with all applicable regulations and standards.
#
# Use at your own risk.

#Carefully read the Read Me to ensure save and more easy employement of the script

import numpy as np
import os
import pydicom as dicom
from scipy import ndimage
from surface_distance import metrics
from datetime import datetime, timedelta
import csv

def remove_line(date=''):
    """
    This function deletes the - from a string with - in it. Used for dates

    input:
    - date: string representing a date

    output:
    - date_string: string representing the same date, but without -'s
    """
    splits = date.split('-')
    splits.reverse()
    date_string=''
    for split in splits:
        date_string += split
    return date_string

def get_dates_between(start_date_str, end_date_str):
    """
    Get a list of dates between two date strings (including start and end dates).
    Args:
        start_date_str (str): Start date in 'DD-MM-YYYY' format.
        end_date_str (str): End date in 'DD-MM-YYYY' format.
    Returns:
        list: List of date strings in 'DD-MM-YYYY' format.
    """
    date_format = '%d-%m-%Y'
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)

    # Generate dates between start and end (inclusive)
    date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    # Convert dates back to string format
    date_strings = [date.strftime(date_format) for date in date_list]

    return date_strings

def is_valid_date(date_string):
    """
    This function checks whether a given date string is valid according to the format DD-MM-YYYY.

    input:
    - date_string: string representing a date in the format 'DD-MM-YYYY'

    output:
    - True if the date_string is a valid date in the specified format
    - False if the format is incorrect or the date is invalid (e.g. 31-02-2023)
    """
    try:
        datetime.strptime(date_string, '%d-%m-%Y')
        return True
    except ValueError:
        return False
    
def read_csv_column(file_path):
    """
    This function reads the first column from a CSV file and returns its values as a list.

    input:
    - file_path: string representing the path to the CSV file

    output:
    - column_data: list containing all values from the first column of the CSV file
    """
    column_data = []

    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # Assuming there's only one column
            column_data.append(row[0])

    return column_data

def get_roi_names(contour_data):
    """
    This function will return the names of different contour data, 
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the 
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names

def get_contours(ds):
    """
    This function reads in the contour points from a rt_struct.dcm file

    input:
    - ds: path to rt_struct.dcm (str)

    output:
    - coordinates_of_contours: list containing the contour points per available contour. 
                               Example: coordinates_of_contours[0] gives all contour coordinates of structure 0
    - not_empty: a list with the location of non empty strucutres in coordinates_of_contours.
    """
    not_empty = []
    # Iterate over each ROI contour
    coordinates_of_contours = []
    check = True
    if 'ROIContourSequence' in ds:
            # Get ROI contours sequence
            roi_contours_sequence = ds.ROIContourSequence
 
            for i, roi_contour in enumerate(roi_contours_sequence, 1):
                
                # Get contour sequence
                if 'ContourSequence' in roi_contour:
                    not_empty.append(int(i-1))
                    contour_sequence = roi_contour.ContourSequence
                    
                    # Iterate over each contour
                    contours_roi = []
                    for j, contour in enumerate(contour_sequence, 1):
                        coords = []                    
                        # Extract contour data points
                        contour_data = contour.ContourData
                        num_points = len(contour_data) // 3  # Each point is represented by triplet (x, y, z)
                        
                        # Print contour data points
                        for k in range(num_points):
                            x, y, z = contour_data[k*3:k*3+3]  # Extract x, y, z coordinates
                            coords.append([x,y,z])
                        contours_roi.append(coords)
                    coordinates_of_contours.append(contours_roi)
                else:
                    coordinates_of_contours.append(0)
    else:
        print("No ROI contours sequence found in the DICOM file.")
        check = False
    return coordinates_of_contours,not_empty,check

def coordinates_to_pixels(structure,xdim,ydim,zdim):
    """
    This function transforms the contour points of a structure to a voxel location instead of a mm distance from a center point

    input:
    - structure: a list of coordinate points of a single structure. Example: coordinates_of_contours[0]
    - xdim, ydim, zdim: floats representing the size of a voxel in the x, y and z direction.

    output:
    - structure_pixels: a list of pixel coordinates representing a structures contour
    """ 
    if structure ==0:
        structure_pixels = 0
    else:
        structure_pixels = []
        for slice in structure:
            x_pixels = np.round(np.transpose(slice)[0]/xdim)
            y_pixels = np.round(np.transpose(slice)[1]/ydim)
            z_pixels = np.round(np.transpose(slice)[2]/zdim)
            slice_pixels = np.transpose([x_pixels,y_pixels,z_pixels])
            structure_pixels.append(slice_pixels)
    return structure_pixels

def find_edges(mask):
    """
    This function finds the edges of the mask

    input:
    - mask: A 3D binary mask (list)

    output:
    - all_edges: edges of the binary mask
    """
    edges = []

    # Find edges along x-axis
    x_edges = np.any(mask[:][:][:], axis=(1, 2))
    edges.append(x_edges)

    # Find edges along y-axis
    y_edges = np.any(mask[:][:][:], axis=(0, 2))
    edges.append(y_edges)

    # Find edges along z-axis
    z_edges = np.any(mask[:][:][:], axis=(0, 1))
    edges.append(z_edges)

    # Find bottom and top sides for each edge
    all_edges = []
    for i, edge in enumerate(edges):
        bottom_side = np.where(edge)[0][0]
        top_side = np.where(edge)[0][-1]
        all_edges.append((bottom_side, top_side))

    return all_edges

def get_cropped_dims(edges_1,edges_2):
    """
    This function finds the minimal dimension the 3D binary mask could have to shrink the amount of datapoints in the mask

    input:
    - mask: A 3D binary mask (list)

    output:
    - xmin,xmax,ymin,ymax,zmin,zmax: indicating the edges of the mask
    """
    if edges_1[0][0] >= edges_2[0][0]:
        xmin = edges_2[0][0]
    else:
        xmin = edges_1[0][0]

    if edges_1[0][1] >= edges_2[0][1]:
        xmax = edges_1[0][1]
    else:
        xmax = edges_2[0][1]

    if edges_1[1][0] >= edges_2[1][0]:
        ymin = edges_2[1][0]
    else:
        ymin = edges_1[1][0]

    if edges_1[1][1] >= edges_2[1][1]:
        ymax = edges_1[1][1]
    else:
        ymax = edges_2[1][1]

    if edges_1[2][0] >= edges_2[2][0]:
        zmin = edges_2[2][0]
    else:
        zmin = edges_1[2][0]
        
    if edges_1[2][1] >= edges_2[2][1]:
        zmax = edges_1[2][1]
    else:
        zmax = edges_2[2][1]
    return xmin, xmax, ymin, ymax, zmin, zmax

def get_min_max(struct):
    """
    This function calculates the minimum and maximum coordinates (x, y, z) from a list of 3D points.

    input:
    - struct: list of 3D coordinates, where each element is a tuple or list like (x, y, z)

    output:
    - struct_xmin: minimum x-coordinate
    - struct_xmax: maximum x-coordinate
    - struct_ymin: minimum y-coordinate
    - struct_ymax: maximum y-coordinate
    - struct_zmin: minimum z-coordinate
    - struct_zmax: maximum z-coordinate
    """
    # A function that get the xmin,xmax,ymin,ymax,zmin,zmax
    struct_xmin = struct[0][0]
    struct_xmax = struct[0][0]
    struct_ymin = struct[0][1]
    struct_ymax = struct[0][1]
    struct_zmin = struct[0][2]
    struct_zmax = struct[0][2]
    for coord in struct:
        if coord[0]>struct_xmax:
            struct_xmax = coord[0]
        elif coord[0]<=struct_xmin:
            struct_xmin = coord[0]
        if coord[1]>struct_ymax:
            struct_ymax = coord[1]
        elif coord[1]<=struct_ymin:
            struct_ymin = coord[1]
        if coord[2]>struct_zmax:
            struct_zmax = coord[2]
        elif coord[2]<=struct_zmin:
            struct_zmin = coord[2]
    return struct_xmin, struct_xmax, struct_ymin,struct_ymax,struct_zmin,struct_zmax

def combined_min_max(AS_struct, MS_struct):
    """
    This function calculates the combined minimum and maximum coordinates (x, y, z)
    from two separate 3D structures.

    input:
    - AS_struct: list of 3D coordinates for the first structure (e.g., automatic segmentation)
    - MS_struct: list of 3D coordinates for the second structure (e.g., manual segmentation)

    output:
    - xmin: minimum x-coordinate from both structures
    - xmax: maximum x-coordinate from both structures
    - ymin: minimum y-coordinate from both structures
    - ymax: maximum y-coordinate from both structures
    - zmin: minimum z-coordinate from both structures
    - zmax: maximum z-coordinate from both structures
    """

    # Get min/max for each structure individually
    as_xmin, as_xmax, as_ymin, as_ymax, as_zmin, as_zmax = get_min_max(AS_struct)
    ms_xmin, ms_xmax, ms_ymin, ms_ymax, ms_zmin, ms_zmax = get_min_max(MS_struct)

    # Compute the overall bounding box by comparing individual bounds
    xmin = as_xmin if as_xmin <= ms_xmin else ms_xmin
    ymin = as_ymin if as_ymin <= ms_ymin else ms_ymin
    zmin = as_zmin if as_zmin <= ms_zmin else ms_zmin
    xmax = as_xmax if as_xmax >= ms_xmax else ms_xmax
    ymax = as_ymax if as_ymax >= ms_ymax else ms_ymax
    zmax = as_zmax if as_zmax >= ms_zmax else ms_zmax

    return xmin, xmax, ymin, ymax, zmin, zmax

def shift_whole_structure(flat_as, flat_ms):
    """
    This function transforms the contour points to a binary 3d contour

    input:
    - pixels: list of contour pixel points of a single structure. Example: structure_pixels

    output:
    - empty_structures: a binary 3D list that shows the contours of a structure
    
    """
    if flat_as != 0:
        xmin, xmax, ymin, ymax, zmin, zmax = combined_min_max(flat_as,flat_ms)
        delta_x = int(abs(xmin-xmax))
        delta_y = int(abs(ymin-ymax))
        delta_z = int(abs(zmin-zmax))
        empty_AS = np.zeros((delta_x+20,delta_y+20,delta_z+20))
        empty_MS = np.zeros((delta_x+20,delta_y+20,delta_z+20))
        for triplet in flat_as:
            empty_AS[int(triplet[0]-xmin+10)][int(triplet[1]-ymin+10)][int(triplet[2]-zmin+10)]=1
        for triplet in flat_ms:
            empty_MS[int(triplet[0]-xmin+10)][int(triplet[1]-ymin+10)][int(triplet[2]-zmin+10)]=1
    else:
        print('structure is empty')
    return empty_AS, empty_MS

def connect_points_3d(point1, point2):
    """
    Connect two 3D points with integer coordinates using a Bresenham-like algorithm.

    Args:
        point1 (list or tuple): The starting point [x1, y1, z1].
        point2 (list or tuple): The ending point [x2, y2, z2].

    Returns:
        list: A list of [x, y, z] points connecting the two input points.
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    # Calculate the deltas
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    # Determine the direction of the steps
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1
    
    # Initialize the starting point
    x, y, z = x1, y1, z1
    
    # Store the resulting points
    points = [[x, y, z]]
    
    # Determine the dominant direction
    if dx >= dy and dx >= dz:  # x is the dominant direction
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x != x2:
            x += sx
            if p1 >= 0:
                y += sy
                p1 -= 2 * dx
            if p2 >= 0:
                z += sz
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append([x, y, z])
    elif dy >= dx and dy >= dz:  # y is the dominant direction
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y != y2:
            y += sy
            if p1 >= 0:
                x += sx
                p1 -= 2 * dy
            if p2 >= 0:
                z += sz
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append([x, y, z])
    else:  # z is the dominant direction
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z != z2:
            z += sz
            if p1 >= 0:
                y += sy
                p1 -= 2 * dz
            if p2 >= 0:
                x += sx
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append([x, y, z])
    
    return points[1:-1]

def fill_holes(pixels):
    """
    This function fills in gaps between non-contiguous 3D points across slices in a 3D structure.

    input:
    - pixels: a list of 2D coordinate slices (each slice is a list of [x, y, z] points),
              representing a 3D structure segmented slice-by-slice

    output:
    - flat: a flattened list of all original and interpolated coordinates (as [x, y, z]),
            where gaps between points in each slice are filled using interpolation
    """
    added_list=[]
    for z in range(len(pixels)):
        slice  = pixels[z]
        diff_array = np.diff(slice,axis=0)
        for i in range(len(diff_array)):
            if np.abs(diff_array[i])[0]>1 or np.abs(diff_array[i])[1]>1:
                added_list.append(connect_points_3d(slice[i],slice[i+1]))
        if np.abs(slice[0]-slice[-1])[0]>1 or np.abs(slice[0]-slice[-1])[1]>1:
                added_list.append(connect_points_3d(slice[0],slice[-1]))
    flat = []
    for slice in pixels:
        for coords in slice:
            flat.append(coords)
    for slice in added_list:
        for coords in slice:
            flat.append(coords)
    return flat


def mask_of_struct_with_complete_fill(structure):
    """
    Transforms a 3D binary contour into a 3D binary mask by robustly filling 
    holes in each 2D slice and ensuring all gaps are filled.

    Parameters:
    - structure: a binary 3D NumPy array representing the contours of a structure.

    Returns:
    - masks: a binary 3D mask of the structure.
    """
    # Initialize an empty list to store the masks
    masks = []

    # Loop through each 2D slice along the z-axis
    for i in range(structure.shape[2]):
        # Extract the 2D slice
        edge_array = structure[:, :, i]

        # Close small gaps in the contours (morphological operation)
        closed_edge_array = ndimage.binary_closing(edge_array)

        # Fill holes in the binary image
        filled_mask = ndimage.binary_fill_holes(closed_edge_array)

        # Append the completely filled mask to the list
        masks.append(filled_mask.astype(int))
    
    # Stack all 2D masks into a 3D mask
    filled_structure = np.stack(masks, axis=2)

    return filled_structure

# APL taken from https://github.com/kkiser1/Autosegmentation-Spatial-Similarity-Metrics
def getEdgeOfMask(mask):
    '''
    Computes and returns edge of a segmentation mask
    '''
    # edge has the pixels which are at the edge of the mask
    edge = np.zeros_like(mask)
    
    # mask_pixels has the pixels which are inside the mask of the automated segmentation result
    mask_pixels = np.where(mask > 0)

    for idx in range(0,mask_pixels[0].size):

        x = mask_pixels[0][idx]
        y = mask_pixels[1][idx]
        z = mask_pixels[2][idx]

        # Count # pixels in 3x3 neighborhood that are in the mask
        # If sum < 27, then (x, y, z) is on the edge of the mask
        if mask[x-1:x+2, y-1:y+2, z-1:z+2].sum() < 27:
            edge[x,y,z] = 1
            
    return edge

def AddedPathLength(auto, gt):
    '''
    Returns the added path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for both auto and gt
    2. Count # pixels on the edge of gt that are not in the edge of auto
    '''
    
    # Check if auto and gt have same dimensions. If not, then raise a ValueError
    if auto.shape != gt.shape:
        raise ValueError('Shape of auto and gt must be identical!')

    # edge_auto has the pixels which are at the edge of the automated segmentation result
    edge_auto = getEdgeOfMask(auto)
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(gt)
    
    # Count # pixels on the edge of gt that are on not in the edge of auto
    apl = (edge_gt > edge_auto).astype(int).sum()
    
    return apl, edge_auto, edge_gt

def get_scores(data_path = '',start_date='28-08-2024',end_date='',xdim=1.17,ydim=1.17,zdim=3.0,exclusion_path=''):
    # Validate date inputs
    if len(start_date) != 10 or len(end_date) != 10:
        print("Dates must be in DD-MM-YYYY format.")
        return "Dates must be in DD-MM-YYYY format."

    if not is_valid_date(start_date) or not is_valid_date(end_date):
        print("Invalid date format.")
        return "Invalid date."

    start_num = int(remove_line(start_date))
    end_num = int(remove_line(end_date))
    if end_num < start_num:
        print("End date must be after start date.")
        return "End date must be after start date."
    #create a list of date strings representing the folders
    date_strings = get_dates_between(start_date,end_date)
    date_list = [remove_line(x) for x in date_strings]

    to_do = []
    for date in date_list:
        day_path = os.path.join(data_path,date)
        if os.path.exists(day_path):
        #load the data for the patients in that day
            for pat_string in os.listdir(day_path):
                pat_path = os.path.join(day_path,pat_string)
                for file in os.listdir(pat_path):
                    #NOTE this part is highly dependent on the naming of your CS and DLS structs. In our institute AS referred to the DLS and MS to the CS
                    #After the AS or MS some specifications are given to match the pairs.
                    if 'MS' in file:
                        splits = file.split('%')
                        part_of_as_name =  f'A{splits[0][1]}%{splits[1]}%{splits[2]}%'
                        scores_name = f'scores_{splits[1]}_{splits[2]}.npy'
                        for as_file in os.listdir(pat_path):
                            if part_of_as_name in as_file:
                                if os.path.exists(os.path.join(pat_path,scores_name)) is False:
                                        to_do.append((os.path.join(pat_path,file),os.path.join(pat_path,as_file)))
    if len(to_do) != 0:
        print(f'TO DO: {int(len(to_do))} PATIENTS')
    else:
        print('Metrics for all patients are already calculated.')
    #If no scores in there create them by loading in the structs
    for patient in to_do:
        loc = np.where(np.array(to_do) == patient)[0][0]
        print(f'PROGRESS: PATIENT {int(loc+1)} of {int(len(to_do))}')
        splits = patient[1].split('%')
        pid_path = patient[1].split('AS')[0][:-1]
        AS_struct = dicom.dcmread(patient[1])
        MS_struct = dicom.dcmread(patient[0])

        # Get all contours coordinates in a list
        AS_contours,AS_filled,AS_check = get_contours(AS_struct)
        MS_contours,MS_filled,MS_check = get_contours(MS_struct)
        matches = []
        if AS_check and MS_check:
            roi_list= read_csv_column(exclusion_path)

            #Get the ROI names and i location in struct file, that are not empty
            AS_ROIs = np.array(get_roi_names(AS_struct))
            MS_ROIs = np.array(get_roi_names(MS_struct))

            #Check if which ROIs are in both AS_strcut and in MS_struct, and get the corresponding location. Only of structures that are in both AS_filled and MS_filled
            
            for i in AS_filled:
                if AS_ROIs[i] in MS_ROIs and AS_ROIs[i] not in roi_list:
                    MS_loc = np.where(MS_ROIs == AS_ROIs[i])[0][0]
                    if MS_loc in MS_filled:
                        matches.append((i,MS_loc))
            scores = []
            for i in range(len(matches)):
                try:
                    if AS_contours[matches[i][0]] == MS_contours[matches[i][1]]:
                        scores.append((AS_ROIs[matches[i][0]],splits[3],str(MS_struct.ReferringPhysicianName),1.0,1.0,0.0,0.0,))
                    else:
                        # Get the pixels of all structures
                        AS_mask = coordinates_to_pixels(AS_contours[matches[i][0]],xdim,ydim,zdim)
                        # Get the pixels of all structures
                        MS_mask = coordinates_to_pixels(MS_contours[matches[i][1]],xdim,ydim,zdim)
                        # Get the shifted contours in pixels
                        flat_as = fill_holes(AS_mask)
                        flat_ms = fill_holes(MS_mask)
                        AS_mask, MS_mask = shift_whole_structure(flat_as,flat_ms)
                        AS_mask= mask_of_struct_with_complete_fill(AS_mask)
                        MS_mask= mask_of_struct_with_complete_fill(MS_mask)
                        #The empty masks are now really big, whilst some structures are really small
                        AS_edges = find_edges(AS_mask)
                        MS_edges = find_edges(MS_mask)

                        #get cropping edges
                        xmin, xmax, ymin, ymax, zmin, zmax = get_cropped_dims(AS_edges,MS_edges)

                        #crop the images according to the maximum dimensions of the structure of the combined masks
                        AS_mask = np.array(AS_mask,dtype=np.bool_)[xmin-1:xmax+1,ymin-1:ymax+1,zmin-1:zmax+1]
                        MS_mask = np.array(MS_mask,dtype=np.bool_)[xmin-1:xmax+1,ymin-1:ymax+1,zmin-1:zmax+1]

                        surface_distances = metrics.compute_surface_distances(AS_mask,MS_mask,(xdim,ydim,zdim))

                        #Metrics'
                        vdsc = metrics.compute_dice_coefficient(AS_mask,MS_mask)
                        hd95 = metrics.compute_robust_hausdorff(surface_distances,95)
                        sdsc = metrics.compute_surface_dice_at_tolerance(surface_distances,3)
                        apl,_,_ = AddedPathLength(AS_mask,MS_mask)
                        scores.append((AS_ROIs[matches[i][0]],splits[3],str(MS_struct.ReferringPhysicianName),vdsc,sdsc,hd95,apl))
                except Exception as e:
                    error_message = str(e)  # Capture the error message
                    print(f'Error occurred for {patient} in {AS_ROIs[matches[i][0]]}: {error_message}')
                    
                    # Create a list that includes patient, roi, and error message
                    List = [splits[3],splits[1], AS_ROIs[matches[i][0]], error_message]
                    
                    # Create a file object for the CSV file
                    with open('Patient_errors.csv', 'a', newline='') as f_object:
                        # Create a CSV writer object with explicit comma delimiter
                        writer_object = csv.writer(f_object, delimiter=';')
                        
                        # Write the List to the CSV file
                        writer_object.writerow(List)
                        
                    # No need to close the file object explicitly, 'with' does it automatically
                    pass
            print(scores)
            np.save(os.path.join(pid_path,f'scores_{splits[1]}_{splits[2]}.npy'),scores)
    return 'All patients are scored'


