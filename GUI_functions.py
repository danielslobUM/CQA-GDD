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
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from csv import writer
from datetime import datetime, timedelta
import json
from calculator_functions import remove_line, is_valid_date, get_dates_between


def check_day_folder_format(day_string=''):
    """
    Function that checks whether the folders in data_path are folders in the day format and thus contain patient or are different type of folders.

    Input: 
    - day_string: string that is the name of a folder

    Output: True or False, depending on the format of the folder. It should be in the YYYYMMDD format. In other words of length 8 and represnting an integer.
    """
    if len(day_string)==8 and day_string.isdigit():
        return True
    else:
        return False

def days_since_first_date(date_array):
    """
    This function determines the days that have passed for each date in the array, corresponding to the first date in the array.

    input:
    - date_array: list of floats that represents dates. Example: 20240423 --> 23-04-2024

    output:
    - diferences_array: np.array that has the same shape as the input array, but then it has the days that have passed
    """
    date_array = np.array(date_array)
    # Convert floats to datetime objects
    dates = [datetime.strptime(str(int(date)), '%Y%m%d') for date in date_array.flatten()]
    
    # Find the first date
    first_date = min(dates)
    
    # Calculate the differences in days
    differences = [(date - first_date).days for date in dates]
    
    # Convert differences to numpy array
    differences_array = np.array(differences).reshape(date_array.shape)
    
    return differences_array

def dates_from_passed_days(date_array, start_date):
    """
    This function computes the calendar dates corresponding to a number of days passed from a given start date.

    input:
    - date_array: list or array of integers/floats that represent days passed since the start_date
    - start_date: float or int that represents a reference date in the format YYYYMMDD. Example: 20240423 --> 23-04-2024

    output:
    - dates_array: np.array with the same shape as the input array, containing date strings in the format DD-MM-YYYY
    """
    # Ensure date_array is a numpy array
    date_array = np.array(date_array)
    
    # Convert start_date to datetime object
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    
    # Calculate the corresponding dates
    dates = [(start_date + timedelta(days=int(days))).strftime('%d-%m-%Y') for days in date_array.flatten()]
    
    # Convert dates to numpy array
    dates_array = np.array(dates).reshape(date_array.shape)
    
    return dates_array

def moving_average_with_outlier_removal(x, y, window_size, threshold=float('inf')):
    """
    Removes outliers from x and y, then computes the moving average for the cleaned data.
    
    Args:
        x (list or np.ndarray): The list of x values (e.g., days_passed).
        y (list or np.ndarray): The list of y values (e.g., np.transpose(roi_vdsc)[0]).
        window_size (int): The size of the moving average window.
        threshold (float): The IQR multiplier to define outliers (default: 1.5).
        
    Returns:
        tuple: Two lists representing the smoothed x and y values.
    """
    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")
    
    # Step 1: Remove outliers
    def remove_outliers(x, y, threshold):
        x = np.array(x)
        y = np.array(y)
        
        # Compute IQR for y
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        # Mask to filter out outliers in y
        non_outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        
        # Filter x and y based on the mask
        filtered_x = x[non_outlier_mask]
        filtered_y = y[non_outlier_mask]
        
        return filtered_x, filtered_y
    
    cleaned_x, cleaned_y = remove_outliers(x, y, threshold)
    
    # Step 2: Compute moving average
    if window_size < 1 or window_size > len(cleaned_x):
        raise ValueError("Window size must be between 1 and the length of the cleaned data.")
    
    smoothed_x = np.convolve(cleaned_x, np.ones(window_size)/window_size, mode='valid')
    smoothed_y = np.convolve(cleaned_y, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed_x.tolist(), smoothed_y.tolist()

def date_difference(date1, date2):
    """
    This function calculates the absolute difference in days between two given dates.

    input:
    - date1: int or float representing a date in the format YYYYMMDD. Example: 20240828 --> 28-08-2024
    - date2: int or float representing a date in the format YYYYMMDD

    output:
    - difference: integer representing the number of days between the two dates
    """
    # Ensure input is integer (handles float cases like 20240828.0)
    date1 = int(date1)
    date2 = int(date2)
    
    # Convert integers to string and then to datetime objects
    d1 = datetime.strptime(str(date1), "%Y%m%d")
    d2 = datetime.strptime(str(date2), "%Y%m%d")
    
    # Compute the difference in days
    return abs((d2 - d1).days)


def make_plot(ROIs=[],start_date='',end_date='',data_path=''):
    # Validate date inputs
    if len(start_date) != 10 or len(end_date) != 10:
        print("Dates must be in YYYY-MM-DD format.")
        return

    if not is_valid_date(start_date) or not is_valid_date(end_date):
        print("Invalid date format.")
        return

    start_num = int(remove_line(start_date))
    end_num = int(remove_line(end_date))
    if end_num < start_num:
        print("End date must be after start date.")
        return

    #create a list of date strings representing the folders
    date_strings = get_dates_between(start_date,end_date)
    date_list = [remove_line(x) for x in date_strings]

    # Load all patient scores for each date
    all_scores = []
    for date in date_list:
        day_path = os.path.join(data_path,date)
        if os.path.exists(day_path):
            patient_list= os.listdir(day_path)
            day_scores = []
            for patient in patient_list:
                patient_path = os.path.join(day_path,patient)
                for file in os.listdir(patient_path):
                    if 'scores' in file:
                        day_scores.append(np.load(os.path.join(patient_path,file)))
            all_scores.append(day_scores)

    # Prepare data containers for each metric
    all_vdsc, all_sdsc, all_hd95, all_apl, all_not_changed = [], [], [], [], []

    # Process each ROI
    for OAR in ROIs:
        vdsc, sdsc, hd95, apl = [], [], [], []
        not_changed = 0
        for patient_scores in all_scores:
            for patient in patient_scores:
                for score in patient:
                    if score[0] == OAR:
                        vals = [float(score[3]), float(score[4]), float(score[5]), float(score[6])]
                        if ('Breast' in OAR or 'CTVp' in OAR) and vals[0] < 0.01 and vals[1] < 0.01 and vals[2] > 50:
                            pass
                            continue
                        vdsc.append([vals[0], score[1]])
                        sdsc.append([vals[1], score[1]])
                        hd95.append([vals[2], score[1]])
                        apl.append([vals[3], score[1]])
                        if vals == [1.0, 1.0, 0.0, 0.0]:
                            not_changed += 1
        all_vdsc.append(vdsc)
        all_sdsc.append(sdsc)
        all_hd95.append(hd95)
        all_apl.append(apl)
        all_not_changed.append(not_changed)

    #Start the plot
    plt.ioff()
    fig, axs = plt.subplots(4,1,figsize=(10,8))
    maxday=0
    vdsc_upper , vdsc_lower = 0.0, 1.0
    sdsc_upper ,sdsc_lower = 0.0, 1.0
    hd95_lower , hd95_upper=100000.0, 0.0
    apl_lower , apl_upper= 1000000.0, 0.0

    #Check which ROIs are required
    for i in range(len(all_vdsc)):
        roi_vdsc = np.array(all_vdsc[i],dtype=float)
        roi_sdsc = np.array(all_sdsc[i],dtype=float)
        roi_hd95 = np.array(all_hd95[i],dtype=float)
        roi_apl = np.array(all_apl[i],dtype=float)
        roi_not_changed = np.array(all_not_changed[i],dtype=int)
        #Check if there are any values

        if np.sum(roi_vdsc[:, 0]) == 0:
            print(f"No patients with {ROIs[i]}.")
            continue

        x = roi_vdsc[:, 1]
        days_passed = days_since_first_date(x)
        first_day_diff = date_difference(int(min(date_list)), min(x))
        if first_day_diff > 0:
            days_passed += first_day_diff

        #Set y lim for vdsc    
        roi_vdsc_mean = round(np.mean(np.transpose(roi_vdsc)[0]),2)
        roi_vdsc_std = round(np.std(np.transpose(roi_vdsc)[0]),2)
        if roi_vdsc_mean + roi_vdsc_std >= vdsc_upper:
            if roi_vdsc_mean + roi_vdsc_std >= 1.0:
                vdsc_upper = 1.02
            else:
                vdsc_upper = roi_vdsc_mean + roi_vdsc_std
        if roi_vdsc_mean - roi_vdsc_std <= vdsc_lower:
            if roi_vdsc_mean - roi_vdsc_std <= 0.0:
                vdsc_lower = -0.02
            else:
                vdsc_lower = roi_vdsc_mean - roi_vdsc_std
        #Set y lim for sdsc  
        roi_sdsc_mean = round(np.mean(np.transpose(roi_sdsc)[0]),2)
        roi_sdsc_std = round(np.std(np.transpose(roi_sdsc)[0]),2) 
        if roi_sdsc_mean + roi_sdsc_std >= sdsc_upper:
            if roi_sdsc_mean + roi_sdsc_std >= 1.0:
                sdsc_upper = 1.02
            else:
                sdsc_upper = roi_sdsc_mean + roi_sdsc_std
        if roi_sdsc_mean - roi_sdsc_std <= sdsc_lower:
            if roi_sdsc_mean - roi_sdsc_std <= 0.0:
                sdsc_lower = -0.02
            else:
                sdsc_lower = roi_sdsc_mean - roi_sdsc_std   
        #Set y lim for hd95 
        roi_hd95_mean = round(np.mean(np.transpose(roi_hd95)[0]),2)
        roi_hd95_std = round(np.std(np.transpose(roi_hd95)[0]),2) 
        if roi_hd95_mean + roi_hd95_std >= hd95_upper:
            hd95_upper =  roi_hd95_mean + roi_hd95_std
        if roi_hd95_mean - roi_hd95_std <= hd95_lower:
            if  roi_hd95_mean - roi_hd95_std <= 0.0:
                hd95_lower = -1.0
            else:
                hd95_lower =  roi_hd95_mean - roi_hd95_std  
        #Set y lim for apl  
        roi_apl_mean = round(np.mean(np.transpose(roi_apl)[0]),2)
        roi_apl_std = round(np.std(np.transpose(roi_apl)[0]),2) 
        if roi_apl_mean + roi_apl_std >= apl_upper:
            apl_upper =  roi_apl_mean + roi_apl_std
        if roi_apl_mean - roi_apl_std <= apl_lower:
            if  roi_apl_mean - roi_apl_std <= 0.0:
                apl_lower = -1.0
            else:
                apl_lower =  roi_apl_mean - roi_apl_std

        #Determine the moving average, if possible
        if len(np.transpose(roi_vdsc)[0])>=40:
            window = 30    
            smooth_x_vdsc , smooth_y_vdsc = moving_average_with_outlier_removal(x= days_passed,y= np.transpose(roi_vdsc)[0],window_size=window,threshold=20.0)
            smooth_x_sdsc , smooth_s_vdsc = moving_average_with_outlier_removal(x= days_passed,y= np.transpose(roi_sdsc)[0],window_size=window,threshold=10.0)
            smooth_x_hd95 , smooth_y_hd95 = moving_average_with_outlier_removal(x= days_passed,y= np.transpose(roi_hd95)[0],window_size=window,threshold=10.0)
            smooth_x_apl , smooth_y_apl = moving_average_with_outlier_removal(x= days_passed,y= np.transpose(roi_apl)[0],window_size=window,threshold=10.0) 
            # Plot the results
            # Check if only 1 ROI, if so plot the SPC plot with control limits                                
            if len(ROIs) == 1:
                axs[0].plot(smooth_x_vdsc , smooth_y_vdsc,color='black',label=f'{ROIs[i]}: {roi_vdsc_mean} +- {roi_vdsc_std} ({round(roi_not_changed/len(roi_vdsc)*100,1)}%, N={len(np.transpose(roi_vdsc)[0])})')
                axs[1].plot(smooth_x_sdsc , smooth_s_vdsc,color='black',label=f'{ROIs[i]}: {roi_sdsc_mean} +- {roi_sdsc_std} ({round(roi_not_changed/len(roi_sdsc)*100,1)}%, N={len(np.transpose(roi_sdsc)[0])})')
                axs[2].plot(smooth_x_hd95 , smooth_y_hd95,color='black',label=f'{ROIs[i]}: {roi_hd95_mean} +- {roi_hd95_std} ({round(roi_not_changed/len(roi_hd95)*100,1)}%, N={len(np.transpose(roi_hd95)[0])})')
                axs[3].plot(smooth_x_apl , smooth_y_apl,color='black',label=f'{ROIs[i]}: {roi_apl_mean} +- {roi_apl_std} ({round(roi_not_changed/len(roi_apl)*100,1)}%, N={len(np.transpose(roi_apl)[0])})')
            #If not 1 ROI, no SPC plot
            else:
                axs[0].plot(smooth_x_vdsc , smooth_y_vdsc,label=f'{ROIs[i]}: {roi_vdsc_mean} +- {roi_vdsc_std} ({round(roi_not_changed/len(roi_vdsc)*100,1)}%, N={len(np.transpose(roi_vdsc)[0])})')
                axs[1].plot(smooth_x_sdsc , smooth_s_vdsc,label=f'{ROIs[i]}: {roi_sdsc_mean} +- {roi_sdsc_std} ({round(roi_not_changed/len(roi_sdsc)*100,1)}%, N={len(np.transpose(roi_sdsc)[0])})')
                axs[2].plot(smooth_x_hd95 , smooth_y_hd95,label=f'{ROIs[i]}: {roi_hd95_mean} +- {roi_hd95_std} ({round(roi_not_changed/len(roi_hd95)*100,1)}%, N={len(np.transpose(roi_hd95)[0])})')
                axs[3].plot(smooth_x_apl , smooth_y_apl,label=f'{ROIs[i]}: {roi_apl_mean} +- {roi_apl_std} ({round(roi_not_changed/len(roi_apl)*100,1)}%, N={len(np.transpose(roi_apl)[0])})')
        #If not enought points for moving average only plot a single point, required for legend
        else:
            axs[0].plot(0,0,label=f'{ROIs[i]}: {roi_vdsc_mean} +- {roi_vdsc_std} ({round(roi_not_changed/len(roi_vdsc)*100,1)}%, N={len(np.transpose(roi_vdsc)[0])})')
            axs[1].plot(0,0,label=f'{ROIs[i]}: {roi_sdsc_mean} +- {roi_sdsc_std} ({round(roi_not_changed/len(roi_sdsc)*100,1)}%, N={len(np.transpose(roi_sdsc)[0])})')
            axs[2].plot(0,0,label=f'{ROIs[i]}: {roi_hd95_mean} +- {roi_hd95_std} ({round(roi_not_changed/len(roi_hd95)*100,1)}%, N={len(np.transpose(roi_hd95)[0])})')
            axs[3].plot(0,0,label=f'{ROIs[i]}: {roi_apl_mean} +- {roi_apl_std} ({round(roi_not_changed/len(roi_apl)*100,1)}%, N={len(np.transpose(roi_apl)[0])})')
        #always plot the scatter points
        #If SPC make these scatter points greay
        if len(ROIs) ==1:
            axs[0].scatter(days_passed,np.transpose(roi_vdsc)[0],color='grey',s=3,alpha=0.45)
            axs[1].scatter(days_passed,np.transpose(roi_sdsc)[0],color='grey',s=3,alpha=0.45)
            axs[2].scatter(days_passed,np.transpose(roi_hd95)[0],color='grey',s=3,alpha=0.45)
            axs[3].scatter(days_passed,np.transpose(roi_apl)[0],color='grey',s=3,alpha=0.45)
        #If not plot them with varying colours
        else:    
            axs[0].scatter(days_passed,np.transpose(roi_vdsc)[0],s=3,alpha=0.45)
            axs[1].scatter(days_passed,np.transpose(roi_sdsc)[0],s=3,alpha=0.45)
            axs[2].scatter(days_passed,np.transpose(roi_hd95)[0],s=3,alpha=0.45)
            axs[3].scatter(days_passed,np.transpose(roi_apl)[0],s=3,alpha=0.45)
    
    # X-axis ticks
    firstday = int(min(date_list))
    days_since_first = days_since_first_date(date_list)
    maxday = max(days_since_first)
    #per day
    if maxday <= 13:
        xdays = np.arange(0, maxday + 1, 1).tolist()
    #per 3 days
    elif maxday <= 34:
        xdays = np.arange(0, maxday, 3).tolist()[:-1] + [maxday]
    #per week
    elif maxday <= 120:
        xdays = np.arange(0, maxday, 7).tolist()[:-1] + [maxday]
    #per month
    elif maxday <=365:
        xdays = np.arange(0, maxday, 30).tolist()[:-1] + [maxday]
    #per 3 months
    else:
        xdays = np.arange(0, maxday, 90).tolist()[:-1] + [maxday]

    dates_list = dates_from_passed_days(xdays, firstday).tolist()

    #Format that is for every plot
    axs[0].set_title('Volumetric Dice Similarity Coefficient',fontsize=12)
    axs[0].set_ylim(vdsc_lower,vdsc_upper)
    axs[0].set_xlim(min(xdays),max(xdays))
    axs[0].set_ylabel('VDSC [-]',fontsize=10)
    axs[0].set_xticks([])
    axs[0].tick_params(axis='y',labelsize=10)

    axs[1].set_title('Surface Dice Similarity Coefficient at 3mm Tolerance',fontsize=12)
    axs[1].set_ylim(sdsc_lower,sdsc_upper)
    axs[1].set_ylabel('SDSC [-]',fontsize=10)
    axs[1].tick_params(axis='y',labelsize=10)
    axs[1].set_xticks([])
    axs[1].set_xlim(min(xdays),max(xdays))

    axs[2].set_title('95th Percentile Hausdorff Distance',fontsize=12)
    axs[2].set_ylabel('HD95 [mm]',fontsize=10)
    axs[2].set_xticks([])
    axs[2].set_ylim(hd95_lower,hd95_upper)
    axs[2].set_xlim(min(xdays),max(xdays))
    axs[2].tick_params(axis='y',labelsize=10)


    axs[3].set_title('Added Path Length',fontsize=12)
    axs[3].set_ylabel(r'APL [N$_{voxels}$]',fontsize=10)
    axs[3].set_ylim(apl_lower,apl_upper)
    axs[3].set_xlim(min(xdays),max(xdays))
    axs[3].tick_params(axis='y',labelsize=10)
    axs[3].set_xticks(xdays)
    axs[3].set_xticklabels(dates_list, rotation = 60,fontsize=10)

    #If there is only 1 ROI selected plot the LCL, UCL and target, and adjust the ylimits accordingly. Only used for SPC plots
    if len(ROIs) == 1:
        roi = ROIs[0]
        with open('control_limits.json', 'r') as file:
            control_limits = json.load(file)
        metrics = ['vdsc','sdsc','hd95','apl']
        for i in range(len(metrics)):
            lcl = control_limits[roi][metrics[i]]['lcl']
            ucl = control_limits[roi][metrics[i]]['ucl']
            target = control_limits[roi][metrics[i]]['target']
            if 'dsc' in metrics[i]:
                axs[i].set_ylim(0,1.0)
            elif metrics[i] == 'hd95':
                axs[i].set_ylim(0,max(np.transpose(roi_hd95)[0])+5.0)
            elif metrics[i] == 'apl':
                axs[i].set_ylim(0,max(np.transpose(roi_apl)[0])+5.0)
            axs[i].axhline(y=target,color='green',alpha=1.0,linestyle='--',linewidth=2,label='Target')
            if ucl != 1.0:
                axs[i].axhline(y=ucl,color='red',alpha=0.5,label='Control Limits')
            if lcl !=0.0:   
                axs[i].axhline(y=lcl,color='red',alpha=0.5,label='Control Limits')
                axs[i].axhspan(0,lcl,facecolor='red',alpha=0.2)
            axs[i].axhspan(lcl,ucl,facecolor='green',alpha=0.2)
            axs[i].axhspan(ucl,10000000.0,facecolor='red',alpha=0.2)
    #plot the legend
    axs[0].legend(loc='center left', bbox_to_anchor = (1,0.5),fontsize=10,fancybox=True)                        
    axs[1].legend(loc='center left', bbox_to_anchor = (1,0.5),fontsize=10,fancybox=True)
    axs[2].legend(loc='center left', bbox_to_anchor = (1,0.5),fontsize=10,fancybox=True)
    axs[3].legend(loc='center left', bbox_to_anchor = (1,0.5),fontsize=10,fancybox=True)
    plt.tight_layout()

    return fig


def get_roi_list(data_path=''):
    """
    This function collects and returns a list of unique ROI (Region of Interest) names 
    from patient score files stored within a nested directory structure.

    input:
    - data_path: string representing the root path where score data is stored. The directory 
      is expected to contain subdirectories for different days, each containing patient folders, 
      which in turn contain numpy '.npy' files with ROI scores.

    output:
    - sorted_str_list: list of unique ROI names (strings), sorted in descending order by 
      how frequently they appear across all patient score files
    """

    all_scores = []
    
    # Traverse all day folders in the provided data path
    for day in os.listdir(data_path):
        if check_day_folder_format(day):  # Skip backup folders
            day_path = os.path.join(data_path, day)
            
            # Traverse all patient folders within the current day folder
            for patient in os.listdir(day_path):
                patient_path = os.path.join(day_path, patient)
                
                # Look for score files within the patient folder
                for file in os.listdir(patient_path):
                    if 'scores' in file:
                        scores = np.load(os.path.join(patient_path, file))
                        
                        # Append score data to all_scores if it's non-empty
                        if len(scores) > 0:
                            all_scores.append(np.load(os.path.join(patient_path, file)))

    # Extract unique ROI names from all patient scores
    roi_list = []
    for patient in all_scores:
        for roi in patient:
            if roi[0] not in roi_list:
                roi_list.append(roi[0])

    if len(roi_list) > 0:
        amount_list = []
        
        # Count how many times each ROI appears in all_scores
        for oar in roi_list:
            amount = 0
            for patient in all_scores:
                for roi in patient:
                    if roi[0] == oar:
                        amount += 1
            amount_list.append(amount)

        # Pair counts with ROI names and sort by frequency (descending)
        sorted_pairs = sorted(zip(amount_list, roi_list), key=lambda x: x[0], reverse=True)

        # Unzip the sorted pairs to retrieve only sorted ROI names
        _, sorted_str_list = zip(*sorted_pairs)

        # Return the list of ROI names sorted by frequency
        sorted_str_list = list(sorted_str_list)
        return sorted_str_list
    else:
        # Return empty list if no ROIs were found
        return roi_list

def make_excel_file(ROIs=[], start_date='', end_date='', data_path='', csv_path=''):
    """
    This function computes summary statistics for specific ROIs (Regions of Interest) over a 
    defined date range and saves the results to a CSV file.

    input:
    - ROIs: list of strings representing ROI names to include in the analysis
    - start_date: string representing the start date in 'YYYY-MM-DD' format
    - end_date: string representing the end date in 'YYYY-MM-DD' format
    - data_path: string representing the path to the root folder containing dated subfolders 
      with patient score files
    - csv_path: string representing the directory path where the CSV file will be saved

    output:
    - None (writes a file named 'BQA_results.csv' to the specified csv_path)
    """

    # Validate date format and logical order
    if len(start_date) != 10 or len(end_date) != 10:
        print("Dates must be in YYYY-MM-DD format.")
        return

    if not is_valid_date(start_date) or not is_valid_date(end_date):
        print("Invalid date format.")
        return

    start_num = int(remove_line(start_date))
    end_num = int(remove_line(end_date))

    if end_num < start_num:
        print("End date must be after start date.")
        return

    # Generate list of dates in compact numeric form (e.g., '20240601')
    date_strings = get_dates_between(start_date, end_date)
    date_list = [remove_line(x) for x in date_strings]

    all_scores = []

    # Load all patient scores from each valid date directory
    for date in date_list:
        day_path = os.path.join(data_path, date)
        if os.path.exists(day_path):
            patient_list = os.listdir(day_path)
            day_scores = []
            for patient in patient_list:
                patient_path = os.path.join(day_path, patient)
                for file in os.listdir(patient_path):
                    if 'scores' in file:
                        day_scores.append(np.load(os.path.join(patient_path, file)))
            all_scores.append(day_scores)

    scores = []

    # For each ROI, extract scores across all dates and compute summary statistics
    for roi in ROIs:
        vdsc = []
        sdsc = []
        hd95 = []
        apl = []
        for day_scores in all_scores:
            for patient in day_scores:
                for values in patient:
                    if values[0] == roi:
                        vdsc.append(float(values[3]))
                        sdsc.append(float(values[4]))
                        hd95.append(float(values[5]))
                        apl.append(float(values[6]))
        # Create a summary row for this ROI
        scores.append([
            roi,
            str(len(vdsc)),
            f'{round(np.mean(vdsc), 2)} +- {round(np.std(vdsc), 2)}',
            f'{round(np.mean(sdsc), 2)} +- {round(np.std(sdsc), 2)}',
            f'{round(np.mean(hd95))} +- {round(np.std(hd95))}',
            f'{round(np.mean(apl))} +- {round(np.std(apl))}'
        ])

    # Define CSV file path and headers
    csv_file = os.path.join(csv_path, 'BQA_results.csv')
    headers = ['ROI', 'Number', 'VDSC', 'SDSC', 'HD95', 'APL']

    # Write headers to CSV file
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(headers)

    # Append each ROI summary row to the CSV
    for row in scores:
        row_to_add = row
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(row_to_add)

def get_data(data_path='',start_date=20240828,end_date=21000101):
    """
    Function that returns the data saved in scores.npy files.

    Input:
    - data_path: string that refers to the folder containing all the day folders
    - start_date: integer that refers to the starting point of the data that needs to be gathered

    Output:
    - ROIs: list of strings of all the unique ROIs that appear in the scores.npy files
    - vdsc: list with all the VDSCs. vdsc[i] correspond to the VDSC of ROIs[i]
    - sdsc: list with all the SDSCs. vdsc[i] correspond to the SDSC of ROIs[i]
    - hd95: list with all the HD95s. vdsc[i] correspond to the HD95 of ROIs[i]
    - apl: list with all the APLs. vdsc[i] correspond to the APL of ROIs[i]
    """

    #Get all unique ROIs available in the score files 
    ROIs = []

    #Go through all the Day folders in data path
    for day in os.listdir(data_path):
        #check if the day folder should be included
        if check_day_folder_format(day) and int(day)>=start_date and int(day)<end_date:
            day_path = os.path.join(data_path,day)
            #Go through all the patients in this day folder
            for patient in os.listdir(day_path):
                patient_path = os.path.join(day_path,patient)
                #check if there is a score file available for that patient
                for file in os.listdir(patient_path):
                    if 'scores' in file:
                        score = np.load(os.path.join(patient_path,file))
                        for OAR in score:
                            #check if this score file has a new unique ROI, if so append it to the ROI list
                            if OAR[0] not in ROIs:
                                ROIs.append(OAR[0])

    #get all the scores in a big list

    #Create empty lists to save the data in per ROI
    vdsc = [[] for _ in range(len(ROIs))]
    sdsc = [[] for _ in range(len(ROIs))]
    hd95 = [[] for _ in range(len(ROIs))]
    apl = [[] for _ in range(len(ROIs))]
    dates = [[] for _ in range(len(ROIs))]
    pids = [[] for _ in range(len(ROIs))]
    #Go through all the day folders in data path
    for day in os.listdir(data_path):
        #check if the day folder should be included
        if check_day_folder_format(day) and int(day)>=start_date:
            day_path = os.path.join(data_path,day)
            #Go through all the patients in this day folder
            for patient in os.listdir(day_path):
                patient_path = os.path.join(day_path,patient)
                #check if there is a score file available for that patient
                for file in os.listdir(patient_path):
                    if 'scores' in file:
                        score = np.load(os.path.join(patient_path,file))
                        for OAR in score:
                            #Get the data per ROI in the correct list and add the new value
                            index = ROIs.index(OAR[0])
                            dates[index].append(float(OAR[1]))
                            vdsc[index].append(float(OAR[3]))
                            sdsc[index].append(float(OAR[4]))
                            hd95[index].append(float(OAR[5]))
                            apl[index].append(float(OAR[6]))
                            pids[index].append(patient)
    return ROIs,dates,vdsc,sdsc,hd95,apl,pids

def nelson_detection(data_path='', limits={},start_date=20250101,value1=2.5,value2=9,value3=6,N_min=10):
    """
    Function that performs the first three adapted Nelson Rules:
    1. If 3 out of 4 metrics are outside of the control limits, it is an outlier.
    2. If 1 out of 4 metrics is above or below the target for 9 or more consecutive points (+-0.5% tolerance) it is a trend shift
    3. If 1 out of 4 metrics is increasing or decreasing for 6 or more consectuvie points it is a trend drift

    Input:
    - data_path: string to the data folder
    - limits: dictionary that contains all the control limits
    - start_date: date where to start the analysis from
    - value1: float that corresponds to the voting system of rule 1
    - value2: integer that corresponds to the amount of consecutive points for rule 2
    - value3: integer that corresponds to the amount of conescutive points for rule 3
    - N_min: value of minimal data points required before determinig if they should be included in Nelson Rules. Same as for calculating control limits

    Output:
    - Rule1: List with outliers according to rule 1
    - Rule2: List with trend shifts according to rule 2
    - Rule3: List with trend drifts according to rule 3
    """
    #Get the data
    rois,dates,vdsc,sdsc,hd95,apl,pids = get_data(data_path=data_path,start_date=start_date)

    #perform rule 1
    list = []
    for i,roi in enumerate(rois):
        if len(vdsc[i])>N_min:
            for j in range(len(vdsc[i])):
                outlier_check = 0.0
                if vdsc[i][j] <limits[roi]['vdsc']['lcl'] or vdsc[i][j] >limits[roi]['vdsc']['ucl']:
                    outlier_check+=1.0
                if sdsc[i][j] < limits[roi]['sdsc']['lcl'] or sdsc[i][j] >limits[roi]['sdsc']['ucl']:
                    outlier_check+=1.0
                if hd95[i][j] < limits[roi]['hd95']['lcl'] or hd95[i][j] >limits[roi]['hd95']['ucl']:
                    outlier_check+=1.0
                if apl[i][j] < limits[roi]['apl']['lcl'] or apl[i][j] >limits[roi]['apl']['ucl']:
                    outlier_check+=1.0
                if outlier_check > value1:
                    list.append([dates[i][j],pids[i][j],roi,vdsc[i][j],sdsc[i][j],hd95[i][j],apl[i][j]])

    Rule1 = [row for row in list]
    Rule1.sort(key=lambda x: (x[0],x[1]))

    #perform rule 2 for VDSC
    list = []
    for i in range(len(rois)):
        roi = rois[i]
        if len(vdsc[i])>N_min:
            above_count = 0
            below_count = 0 
            above_start_index = None
            below_start_index = None

            for j, value in enumerate(vdsc[i]):
                if value > limits[roi]['vdsc']['target']*1.005:
                    if above_count == 0:
                        above_start_index = j
                    above_count += 1
                    if below_count >=value2:
                        # list.append(f'Below Target: {roi} - VDSC - {below_count} points start at i={below_start_index}')
                        list.append([roi,'VDSC','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                elif value < limits[roi]['vdsc']['target']*0.995:
                    if below_count == 0:
                        below_start_index = j
                    below_count += 1
                    if above_count >=value2:
                        # list.append(f'Above Target: {roi} - VDSC - {above_count} points start at i={above_start_index}')
                        list.append([roi,'VDSC','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
                else:
                    if below_count >=value2:
                    # list.append(f'Below Target: {roi} - VDSC - {below_count} points start at i={below_start_index}')
                        list.append([roi,'VDSC','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                    if above_count >=value2:
                    # list.append(f'Above Target: {roi} - VDSC - {above_count} points start at i={above_start_index}')
                        list.append([roi,'VDSC','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
        #perform rule 2 for hd95    
        if len(hd95[i])>N_min:
            above_count = 0
            below_count = 0 
            above_start_index = None
            below_start_index = None

            for j, value in enumerate(hd95[i]):
                if value > limits[roi]['hd95']['target']*1.005:
                    if above_count == 0:
                        above_start_index = j
                    above_count += 1
                    if below_count >=value2:
                        # list.append(f'Below Target: {roi} - HD95 - {below_count} points start at i={below_start_index}')
                        list.append([roi,'HD95','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                elif value < limits[roi]['hd95']['target']*0.995:
                    if below_count == 0:
                        below_start_index = j
                    below_count += 1
                    if above_count >=value2:
                        # list.append(f'Above Target: {roi} - HD95 - {above_count} points start at i={above_start_index}')
                        list.append([roi,'HD95','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
                else:
                    if below_count >=value2:
                    # list.append(f'Below Target: {roi} - VDSC - {below_count} points start at i={below_start_index}')
                        list.append([roi,'HD95','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                    if above_count >=value2:
                    # list.append(f'Above Target: {roi} - VDSC - {above_count} points start at i={above_start_index}')
                        list.append([roi,'HD95','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
        #perform rule 2 for sdsc  
        if len(sdsc[i])>N_min:
            above_count = 0
            below_count = 0 
            above_start_index = None
            below_start_index = None

            for j, value in enumerate(sdsc[i]):
                if value > limits[roi]['sdsc']['target']*1.005:
                    if above_count == 0:
                        above_start_index = j
                    above_count += 1
                    if below_count >=value2:
                        # list.append(f'Below Target: {roi} - SDSC - {below_count} points start at i={below_start_index}')
                        list.append([roi,'SDSC','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                elif value < limits[roi]['sdsc']['target']*0.995:
                    if below_count == 0:
                        below_start_index = j
                    below_count += 1
                    if above_count >=value2:
                        # list.append(f'Above Target: {roi} - SDSC - {above_count} points start at i={above_start_index}')
                        list.append([roi,'SDSC','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
                else:
                    if below_count >=value2:
                    # list.append(f'Below Target: {roi} - SDSC - {below_count} points start at i={below_start_index}')
                        list.append([roi,'SDSC','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                    if above_count >=value2:
                    # list.append(f'Above Target: {roi} - SDSC - {above_count} points start at i={above_start_index}')
                        list.append([roi,'SDSC','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
        #perform rule 2 for apl  
        if len(apl[i])>N_min:
            above_count = 0
            below_count = 0 
            above_start_index = None
            below_start_index = None

            for j, value in enumerate(apl[i]):
                if value > limits[roi]['apl']['target']*1.005:
                    if above_count == 0:
                        above_start_index = j
                    above_count += 1
                    if below_count >=value2:
                        # list.append(f'Below Target: {roi} - APL - {below_count} points start at i={below_start_index}')
                        list.append([roi,'APL','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                elif value < limits[roi]['apl']['target']*0.995:
                    if below_count == 0:
                        below_start_index = j
                    below_count += 1
                    if above_count >=value2:
                        # list.append(f'Above Target: {roi} - APL - {above_count} points start at i={above_start_index}')
                        list.append([roi,'APL','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None
                else:
                    if below_count >=value2:
                    # list.append(f'Below Target: {roi} - APL - {below_count} points start at i={below_start_index}')
                        list.append([roi,'APL','Below Target',int(dates[i][below_start_index]),below_count])
                    below_count = 0
                    below_start_index = None
                    if above_count >=value2:
                    # list.append(f'Above Target: {roi} - APL - {above_count} points start at i={above_start_index}')
                        list.append([roi,'APL','Above Target',int(dates[i][above_start_index]),above_count])
                    above_count = 0
                    above_start_index = None

    Rule2 = [row for row in list]
    Rule2.sort(key=lambda x: x[3])

    #perform rule 3 for vdsc  
    list = []
    for j in range(len(rois)):
        values = vdsc[j]
        if len(values)>N_min:
            increasing_count = 0
            decreasing_count = 0
            roi = rois[j]
            for i in range(1, len(values)):
                if values[i] > values[i - 1]:
                    increasing_count += 1
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} VDSC has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'VDSC','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    decreasing_count = 0
                elif values[i] < values[i - 1]:
                    decreasing_count += 1
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} VDSC has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'VDSC','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    increasing_count = 0
                else:
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} VDSC has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'VDSC','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} VDSC has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'VDSC','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    increasing_count = 0
                    decreasing_count = 0
            #perform rule 3 for hd95  
            increasing_count = 0
            decreasing_count = 0
            roi = rois[j]
            values = hd95[j]
            for i in range(1, len(values)):
                if values[i] > values[i - 1]:
                    increasing_count += 1
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} HD95 has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'HD95','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    decreasing_count = 0
                elif values[i] < values[i - 1]:
                    decreasing_count += 1
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} HD95 has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'HD95','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    increasing_count = 0
                else:
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} HD95 has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'HD95','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    if decreasing_count >=value3:
                        # list.append(f'Decreasing trend: {roi} HD95 has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'HD95','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    increasing_count = 0
                    decreasing_count = 0
            #perform rule 3 for sdsc  
            increasing_count = 0
            decreasing_count = 0
            roi = rois[j]
            values = sdsc[j]
            for i in range(1, len(values)):
                if values[i] > values[i - 1]:
                    increasing_count += 1
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} SDSC has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'SDSC','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    decreasing_count = 0
                elif values[i] < values[i - 1]:
                    decreasing_count += 1
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} SDSC has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'SDSC','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    increasing_count = 0
                else:
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} SDSC has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'SDSC','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} SDSC has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'SDSC','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    increasing_count = 0
                    decreasing_count = 0
            #perform rule 3 for apl 
            increasing_count = 0
            decreasing_count = 0
            roi = rois[j]
            values = apl[j]
            for i in range(1, len(values)):
                if values[i] > values[i - 1]:
                    increasing_count += 1
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} APL has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'APL','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    decreasing_count = 0
                elif values[i] < values[i - 1]:
                    decreasing_count += 1
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} APL has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'APL','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    increasing_count = 0
                else:
                    if increasing_count >= value3:
                        # list.append(f'Increasing trend: {roi} APL has increased for {increasing_count} starting at {i-increasing_count}')
                        list.append([roi,'APL','Increasing Trend',int(dates[j][int(i-increasing_count)]),int(increasing_count)])
                    if decreasing_count >= value3:
                        # list.append(f'Decreasing trend: {roi} APL has decreased for {decreasing_count} starting at {i-decreasing_count}')
                        list.append([roi,'APL','Decreasing Trend',int(dates[j][int(i-decreasing_count)]),int(decreasing_count)])
                    increasing_count = 0
                    decreasing_count = 0
    Rule3 = [row for row in list]
    Rule3.sort(key=lambda x: x[3])

    return Rule1, Rule2, Rule3

def update_csv_1(csv_path, ListA):
    """
    This function updates a CSV file by checking for duplicate entries in the file 
    based on full row match with items from a given list. If a row from the list 
    is not already in the CSV, it will be appended. Made for rule 1 format

    input:
    - csv_path: string representing the full path to the CSV file to be updated
    - ListA: list of entries, where each entry is a list containing 7 values:
        [int, int, str, float, float, float, int]

    output:
    - None (updates the CSV file in place)
    """
    # Step 1: Read existing rows from the CSV file
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file,delimiter=';')
            existing_rows = list(reader)
    except FileNotFoundError:
        # If the file doesn't exist, we'll just work with an empty list.
        existing_rows = []

    # Step 2: Process each entry in ListA
    for item in ListA:
        item_values = [int(item[0]),int(item[1]),str(item[2]),round(float(item[3]),4),round(float(item[4]),4),round(float(item[5]),4),int(item[6])]
        found_match = False

        # Check if any existing row matches the first 4 elements
        for idx, row in enumerate(existing_rows):
            try:
                row_values = [int(row[0]),int(row[1]),str(row[2]),round(float(row[3]),4),round(float(row[4]),4),round(float(row[5]),4),int(row[6])]
                if row_values == item_values:
                    found_match = True
            except:
                found_match = False

        if not found_match:
            # If no match was found or updated, append the new row
            existing_rows.append(item)

    # Step 3: Write the updated rows back to the CSV file
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file,delimiter=';')
        writer.writerows(existing_rows)

def update_csv_23(csv_path, ListA):
    """
    This function updates a CSV file by checking for matching rows based on the first 
    4 columns of each entry. If a match is found but the fifth value (an integer) differs, 
    the row is replaced with the new one. If no match is found, the new row is appended.
    Made for the rule 2 and 3 save format

    input:
    - csv_path: string representing the full path to the CSV file to be updated
    - ListA: list of entries, where each entry is a list in the format:
        ['Item A', 'Item B', 'Item C', Integer1, Integer2]

    output:
    - None (updates the CSV file in place)
    """
    # Step 1: Read existing rows from the CSV file
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file,delimiter=';')
            existing_rows = list(reader)
    except FileNotFoundError:
        # If the file doesn't exist, we'll just work with an empty list.
        existing_rows = []

    # Step 2: Process each entry in ListA
    for item in ListA:
        item_values = item[:4]  # ['Item A', 'Item B', 'Item C', Integer1]
        new_integer2 = item[4]  # Integer2
        found_match = False
        updated = False

        # Check if any existing row matches the first 4 elements
        for idx, row in enumerate(existing_rows):
            row_values = row[:4]  # ['Item A', 'Item B', 'Item C', Integer1]
            try:
                row_values[3] = int(row[3])
                if row_values == item_values:
                    if int(row[4]) == new_integer2:
                        # Row is identical, skip adding
                        found_match = True
                    else:
                        # Replace the row if Integer2 is different
                        existing_rows[idx] = item
                        updated = True
                    break
            except:
                found_match = False
                updated = False

        if not found_match and not updated:
            # If no match was found or updated, append the new row
            existing_rows.append(item)

    # Step 3: Write the updated rows back to the CSV file
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file,delimiter=';')
        writer.writerows(existing_rows)
