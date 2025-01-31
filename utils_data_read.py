import gzip
import csv
import re
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from collections import OrderedDict

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
    """
    Read original dat.gz file and select I-24 MOTION WB portion between startmile and endmile
    write rows into a new csv file in the following format
    | timestamp | milemarker | lane | speed | volume | occupancy |

    Parameters:
    ----------
    file_path : string
        path of the original RDS data in dat.gz
    write_file_path : string
        path of the new csv file to store filtered data
    startmile : float
        starting milemarker to filter e.g., 54.1
    endmile : float
        ending milemarker to filter e.g., 57.6

    Returns: None
    """

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
    rds_file is the processed RDS data, aggregated in 5min
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
    
    detector_data["flow"]=detector_data["volume"]
    detector_data["density"]=detector_data["flow"]/detector_data["speed"]
    return detector_data

def extract_mean_speed_all_lanes(xml_file):
    '''
    given output of lanearea(E2) detectors, extract meanSpeed for all lanes
    lane_speeds[lane_id] = [speeds at each time interval]
    '''
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Dictionary to store mean speeds for each lane
    lane_speeds = {}
    time_intervals = []
    prev_time = -1

    # Iterate over all intervals in the XML file
    for interval in root.findall('interval'):
        lane_id = interval.get('id')
        mean_speed = float(interval.get('meanSpeed'))
        begin_time = float(interval.get("begin"))
        if begin_time != prev_time:
            time_intervals.append(begin_time)
            prev_time = begin_time

        # If lane_id is not already in the dictionary, initialize a new list
        if lane_id not in lane_speeds:
            lane_speeds[lane_id] = []

        # Append the meanSpeed to the corresponding lane's list
        lane_speeds[lane_id].append(mean_speed)

    # Calculate travel time
    travel_time_all_lane = {}
    for lane_id, speeds in lane_speeds.items():
        speeds = np.array(speeds)
        speeds = np.where(speeds == 0, 0.1, speeds) # avoide divide by zero
        tt = 1300/speeds
        travel_time_all_lane[lane_id] = tt

    return lane_speeds, travel_time_all_lane, time_intervals



def parse_and_reorder_xml(xml_file, output_csv, link_names=None):
    '''
    Parse xml file (ordered by timestep) to a csv file (ordered by vehicleID, in NGSIM format)
    'VehicleID', 'Time', 'LaneID', 'LocalY', 'MeanSpeed', 'MeanAccel', 'VehLength', 'VehClass', 'FollowerID', 'LeaderID'
    link_names: selected links that the data will be written (usually to filter mainline only)
    if link_names is set to None, then no data will be filtered (select all links)

    Parameters:
    ----------
    xml_file : string
        path to the fcd xml file generated during run_sumo
    output_csv : string
        path of the new csv file to store the output data
    link_names : 
        None (default): no data will be filtered. Write all data to output_csv
        list : link names specified as a list of strings. Only write data where link_name is in the given list
        dict: {key, val}: write to multiple output_csv files, each append with the key string. Val corresponding to each key is a list of link names. Useful to specified multiple lanes
    Returns: None
    '''
    # OrderedDict to store data by vehicle id, preserving the order of first appearance
    vehicle_data = OrderedDict()
    
    # Stream the XML file with iterparse
    context = ET.iterparse(xml_file, events=('end',))
    
    # Parse each timestep and collect vehicle data
    print("parsing xml file...")
    for event, elem in context:
        if elem.tag == 'timestep':
            time = elem.get('time', '-1')  # Get the time for this timestep
            
            for vehicle in elem.findall('vehicle'):
                vehicle_id = vehicle.get('id', '-1')
                lane_id = vehicle.get('lane', '-1')
                local_y = vehicle.get('x', '-1')
                mean_speed = vehicle.get('speed', '-1')
                mean_accel = vehicle.get('accel', '-1')  # Assuming 'accel' exists
                veh_length = vehicle.get('length', '-1')
                veh_class = vehicle.get('type', '-1')
                follower_id = vehicle.get('pos', '-1')  # Assuming 'pos' is follower ID
                leader_id = vehicle.get('slope', '-1')  # Assuming 'slope' is leader ID
                
                # Ensure vehicle_id is added the first time it appears
                if vehicle_id not in vehicle_data:
                    vehicle_data[vehicle_id] = []
                
                # Append the row for this vehicle at this timestep
                vehicle_data[vehicle_id].append([
                    vehicle_id, time, lane_id, local_y, mean_speed, mean_accel, 
                    veh_length, veh_class, follower_id, leader_id
                ])
            elem.clear()  # Free memory

    # Reorder data by vehicle_id first appearance, then by time
    print("reorder by time...")
    for vehicle_id in vehicle_data:
        vehicle_data[vehicle_id].sort(key=lambda x: float(x[1]))

    # Write the result to a CSV file
    print("writing to csv...")
    multiple_writers = False
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['VehicleID', 'Time', 'LaneID', 'LocalY', 'MeanSpeed', 'MeanAccel', 
                         'VehLength', 'VehClass', 'FollowerID', 'LeaderID'])
        
        # Write the sorted data
        if link_names is None: # write all data
            for vehicle_id in vehicle_data:
                for row in vehicle_data[vehicle_id]:
                    writer.writerow(row)
        elif isinstance(link_names, list) : # when link_names is a list of links, Write selected links data
            for vehicle_id in vehicle_data:
                for row in vehicle_data[vehicle_id]:
                    if row[2] in link_names:
                        writer.writerow(row)
        else:
            multiple_writers = True

    if multiple_writers:
        for key, links in link_names.items():
            csv_name = output_csv.split(".")[0]+"_"+key+".csv"
            with open(csv_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['VehicleID', 'Time', 'LaneID', 'LocalY', 'MeanSpeed', 'MeanAccel', 
                                'VehLength', 'VehClass', 'FollowerID', 'LeaderID'])
                for vehicle_id in vehicle_data:
                    for row in vehicle_data[vehicle_id]:
                        if row[2] in links:
                            writer.writerow(row)
            print(csv_name, " is saved.")

    return



def det_to_csv(xml_file, suffix=""):
    '''
    TO BE REMOVED
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


def filter_trajectory_data(input_file, output_file, start_time, end_time):
    # filter fcd.xml output with specified start_time and end_time
    # Open output file and write the XML header and root opening tag
    time_offset = start_time
    with open(output_file, 'w') as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<fcd-export xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ')
        out.write('xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/fcd_file.xsd">\n')

        # Parse the input file iteratively
        for event, elem in ET.iterparse(input_file, events=("start", "end")):
            if event == "end" and elem.tag == "timestep":
                time = float(elem.attrib["time"])
                
                # Check if the timestep falls within the given range
                if start_time <= time <= end_time:
                    # Adjust the time to start from 0
                    elem.attrib["time"] = f"{time - time_offset:.2f}"
                    # Write the timestep element to the output file
                    out.write(ET.tostring(elem, encoding="unicode"))
                
                # Clear the element from memory to save space
                elem.clear()

        # Close the root tag
        out.write('</fcd-export>\n')

    return

# copied from onramp_calibrate.py
# def write_vehicle_trajectories_to_csv(readfilename, writefilename):
#     # Start SUMO simulation with TraCI
#     traci.start(["sumo", "-c", readfilename+".sumocfg"])
    
#     # Replace "your_routes_file.rou.xml" with the actual path to your SUMO route file
#     route_file_path = readfilename+".rou.xml"
#     # Get a list of vehicle IDs from the route file
#     predefined_vehicle_ids = get_vehicle_ids_from_routes(route_file_path)

#     # Print the list of vehicle IDs
#     print("List of Predefined Vehicle IDs:", predefined_vehicle_ids)


#     # Open the CSV file for writing
#     with open(writefilename, 'w') as csv_file:
#         # Write header
#         # Column 1:	Vehicle ID
#         # Column 2:	Frame ID
#         # Column 3:	Lane ID
#         # Column 4:	LocalY
#         # Column 5:	Mean Speed
#         # Column 6:	Mean Acceleration
#         # Column 7:	Vehicle length
#         # Column 8:	Vehicle Class ID
#         # Column 9:	Follower ID
#         # Column 10: Leader ID

#         csv_file.write("VehicleID, Time, LaneID, LocalY, MeanSpeed, MeanAccel, VehLength, VehClass, FollowerID, LeaderID\n")
#         # vehicle_id = "carflow1.131"
#         # Run simulation steps
#         step = 0
#         while traci.simulation.getMinExpectedNumber() > 0:
#             # Get simulation time
#             simulation_time = traci.simulation.getTime()

#             # Get IDs of all vehicles
#             vehicle_ids = traci.vehicle.getIDList()

#             # Iterate over all vehicles
#             for vehicle_id in vehicle_ids:
#                 # Get vehicle position and speed
#                 position = traci.vehicle.getPosition(vehicle_id)
#                 laneid = traci.vehicle.getLaneID(vehicle_id)
#                 speed = traci.vehicle.getSpeed(vehicle_id)
#                 accel = traci.vehicle.getAcceleration(vehicle_id)
#                 cls = traci.vehicle.getVehicleClass(vehicle_id)

#                 # Write data to the CSV file - similar to NGSIM schema
#                 csv_file.write(f"{vehicle_id} {simulation_time} {laneid} {position[0]} {speed} {accel} {-1} {cls} {-1} {-1}\n")

#             # try to overwite acceleration of one vehicle
#             # if 300< step <400:
#             #     traci.vehicle.setSpeed(vehicle_id, 0)
#             # Simulate one step
#             traci.simulationStep()
#             step += 1

#     # Close connection
#     traci.close()
#     print("Complete!")

#     return


if __name__ == "__main__":

    file_path = r'PATH TO RDS.dat.gz'
    write_file_path = r'data/RDS/I24_WB_52_60_11132023.csv'
    # read_and_filter_file(file_path, write_file_path, 52, 57.5)
    # 
    # vis_rds_color(write_file_path=write_file_path, lane_number=None)
    # plot_ramp_volumes(write_file_path)

    