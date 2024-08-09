import gzip
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d

def extract_mile_marker(link_name):
    "link_name: R3G-00I24-59.7W Off Ramp (280)"
    matches = re.findall(r'-([0-9]+(?:\.[0-9]+)?)', link_name)
    return float(matches[1]) if len(matches) > 1 else None

def extract_lane_number(lane_name):
    match = re.search(r'Lane(\d+)', lane_name)
    return int(match.group(1)) if match else None

def is_i24_westbound_milemarker(link_name, min_mile, max_mile):
    if 'I24' not in link_name or 'W' not in link_name:
        return False
    mile_marker = extract_mile_marker(link_name)
    if mile_marker is None:
        return False
    return min_mile <= mile_marker <= max_mile

def safe_float(value):
    try:
        return float(value)
    except:
        return None

def read_and_filter_file(file_path, write_file_path, startmile, endmile):
    '''
    read original dat.gz file and select I-24 MOTION WB portion
    write rows into a new file
    | timestamp | milemarker | lane | speed | volume | occupancy |
    '''
    selected_fieldnames = ['timestamp', 'link_name', 'milemarker', 'lane', 'speed', 'volume', 'occupancy']
    open_func = gzip.open if file_path.endswith('.gz') else open
    with open_func(file_path, mode='rt') as file:
        reader = csv.DictReader(file)
        with open(write_file_path, mode='w', newline='') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=selected_fieldnames)
            writer.writeheader()
            for row in reader:
                if is_i24_westbound_milemarker(row[' link_name'], startmile, endmile): # 58-63
                    selected_row = {
                        'timestamp': row['timestamp'],
                        'link_name': row[' link_name'],
                        'milemarker': extract_mile_marker(row[' link_name']),
                        'lane': extract_lane_number(row[' lane_name']),
                        'speed': safe_float(row[' speed']),
                        'volume': safe_float(row[' volume']),
                        'occupancy': safe_float(row[' occupancy'])
                    }
                    writer.writerow(selected_row)

def interpolate_zeros(arr):
    arr = np.array(arr)
    interpolated_arr = arr.copy()
    
    for i, row in enumerate(arr):
        zero_indices = np.where(row < 4)[0]
        
        if len(zero_indices) > 0:
            # Define the x values for the valid (non-zero) data points
            x = np.arange(len(row))
            valid_indices = np.setdiff1d(x, zero_indices)
            
            if len(valid_indices) > 1:  # Ensure there are at least two points to interpolate
                # Create the interpolation function based on valid data points
                interp_func = interp1d(x[valid_indices], row[valid_indices], kind='linear', fill_value="extrapolate")
                
                # Replace the zero values with interpolated values
                interpolated_arr[i, zero_indices] = interp_func(zero_indices)
    
    return interpolated_arr

def rds_to_matrix(rds_file, det_locations ):
    '''
    Read RDS data from a CSV file and output a matrix of [N_dec, N_time] size,
    where N_dec is the number of detectors and N_time is the number of aggregated
    time intervals of 5 minutes.
    
    Parameters:
    - rds_file: Path to the RDS data CSV file.
    - det_locations: List of strings representing RDS sensor locations in the format "milemarker_lane", e.g., "56_7_3".
    
    Returns:
    - matrix: A numpy array of shape [N_dec, N_time].

    SUMO lane is 0-indexed (from right), while RDS lanes are 1-index (from left)
    '''
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(rds_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    milemarkers = [round(float(".".join(location.split('_')[:2])),1) for location in det_locations]
    lanes = [int(location.split('_')[-1])+1 for location in det_locations]
    macro_data = {"speed": [], "volume": [], "occupancy": []}

    for milemarker, lane in zip(milemarkers, lanes):
        # Filter rows based on milemarker and lane
        filtered_df = df[(df['milemarker'] == milemarker) & (df['lane'] == lane)]
        
        # Aggregate by 5-minute intervals (assuming 'timestamp' is already in datetime format)
        if filtered_df.empty:
            print(f"No RDS data for milemarker {milemarker} lane {lane}")
        else:
            aggregated = filtered_df.groupby(pd.Grouper(key='timestamp', freq='5min')).agg({
                'speed': 'mean',
                'volume': 'sum',
                'occupancy': 'mean'
            }).reset_index()

            macro_data["speed"].append(aggregated["speed"].values)
            macro_data["volume"].append(aggregated["volume"].values * 12) # convert to vVeh/hr
            macro_data["occupancy"].append(aggregated["occupancy"].values)

    macro_data["speed"] = np.vstack(macro_data["speed"]) # [N_dec, N_time]
    macro_data["volume"] = np.vstack(macro_data["volume"]) # [N_dec, N_time]
    macro_data["occupancy"] = np.vstack(macro_data["occupancy"]) # [N_dec, N_time]

    # postprocessing
    macro_data["volume"] = interpolate_zeros(macro_data["volume"])
    macro_data["flow"] = macro_data["volume"]
    macro_data["density"] = macro_data["flow"]/macro_data["speed"]

    return macro_data

def extract_sim_meas(measurement_locations, file_dir = ""):
    """
    Extract simulated traffic measurements (Q, V, Occ) from SUMO detector output files (xxx.out.xml).
    Q/V/Occ: [N_dec x N_time]
    measurement_locations: a list of strings that map detector IDs
    """
    # Initialize an empty list to store the data for each detector
    detector_data = {"speed": [], "volume": [], "occupancy": []}

    for detector_id in measurement_locations:
        # Construct the filename for the detector's output XML file
        # print(f"reading {detector_id}...")
        filename = os.path.join(file_dir, f"det_{detector_id}.out.xml")
        
        # Check if the file exists
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist. Skipping this detector.")
            continue
        
        # Parse the XML file
        tree = ET.parse(filename)
        root = tree.getroot()

        # Initialize a list to store the measurements for this detector
        speed = []
        volume = []
        occupancy = []

        # Iterate over each interval element in the XML
        for interval in root.findall('interval'):
            # Extract the entered attribute (number of vehicles entered in the interval)
            speed.append(float(interval.get('speed')) * 2.237) # convert m/s to mph
            volume.append(float(interval.get('flow')))
            occupancy.append(float(interval.get('occupancy')))
        
        # Append the measurements for this detector to the detector_data list
        detector_data["speed"].append(speed) # in mph
        detector_data["volume"].append(volume) # in veh/hr
        detector_data["occupancy"].append(occupancy) # in %
    
    for key, val in detector_data.items():
        detector_data[key] = np.array(val)
        # print(val.shape)
    
    return detector_data



def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = []
    for timestep in root.findall('timestep'):
        for vehicle in timestep.findall('vehicle'):
            vehicle_id = vehicle.get('id')
            time = timestep.get('time')
            lane_id = vehicle.get('lane')
            local_y = vehicle.get('x', '-1')
            mean_speed = vehicle.get('speed', '-1')
            mean_accel = vehicle.get('accel', '-1')  # Assuming accel is mean acceleration
            veh_length = vehicle.get('length', '-1')
            veh_class = vehicle.get('type', '-1')
            follower_id = vehicle.get('pos', '-1')  # Assuming pos is follower ID
            leader_id = vehicle.get('slope', '-1')  # Assuming slope is leader ID
            
            row = [vehicle_id, time, lane_id, local_y, mean_speed, mean_accel, veh_length, veh_class, follower_id, leader_id]
            # data.append([str(item) for item in row])
            data.append([" ".join(str(num) for num in row)])
    
    return data

# Function to write data to CSV
def write_csv(data, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['VehicleID', 'Time', 'LaneID', 'LocalY', 'MeanSpeed', 'MeanAccel', 'VehLength', 'VehClass', 'FollowerID', 'LeaderID'])
        # Write rows
        writer.writerows(data)


def det_to_csv(xml_file, suffix=""):
    '''
    Read detector data {DET}.out.xml and re-write them to .csv files with names {DET}{suffix}.csv
    '''

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Open a CSV file for writing
    csv_file_name = xml_file.split(".")[-3]
    with open(f'{csv_file_name}{suffix}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        header = ["begin", "end", "id", "nVehContrib", "flow", "occupancy", "speed", "harmonicMeanSpeed", "length", "nVehEntered"]
        writer.writerow(header)
        
        # Write the data rows
        for interval in root.findall('interval'):
            row = [
                float(interval.get("begin")),
                float(interval.get("end")),
                interval.get("id"),
                int(interval.get("nVehContrib")),
                float(interval.get("flow")),
                float(interval.get("occupancy")),
                float(interval.get("speed")),
                float(interval.get("harmonicMeanSpeed")),
                float(interval.get("length")),
                int(interval.get("nVehEntered"))
            ]
            writer.writerow(row)

    return

def fcd_to_csv_byid(xml_file, csv_file):
    print(f"parsing {xml_file}...")
    data = parse_xml(xml_file)
    print(f"writing {csv_file}...")
    write_csv(data, csv_file)
    return



if __name__ == "__main__":

    file_path = r'PATH TO RDS.dat.gz'
    write_file_path = r'data/RDS/I24_WB_52_60_11132023.csv'
    # read_and_filter_file(file_path, write_file_path, 52, 57.5)
    # vis_rds_lines(write_file_path=write_file_path)
    # vis_rds_color(write_file_path=write_file_path, lane_number=None)
    # plot_ramp_volumes(write_file_path)

    