import traci
import traci.constants as tc
import xml.etree.ElementTree as ET
import os
import sys
import subprocess
import pickle
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import macro
import onramp_calibrate as onramp
import csv

# import pandas as pd

def run_simulation(readfilename):
    
    traci.start(["sumo", "-c", readfilename+".sumocfg"])

    # Run simulation steps
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()


def get_vehicle_ids_from_routes(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot()

    vehicle_ids = []
    for route in root.findall('.//vehicle'):
        vehicle_id = route.get('id')
        vehicle_ids.append(vehicle_id)

    return vehicle_ids





def write_vehicle_trajectories_to_csv(readfilename, writefilename):
    # Start SUMO simulation with TraCI
    traci.start(["sumo", "-c", readfilename+".sumocfg"])
    
    # Replace "your_routes_file.rou.xml" with the actual path to your SUMO route file
    route_file_path = readfilename+".rou.xml"
    # Get a list of vehicle IDs from the route file
    predefined_vehicle_ids = get_vehicle_ids_from_routes(route_file_path)

    # Print the list of vehicle IDs
    print("List of Predefined Vehicle IDs:", predefined_vehicle_ids)

    
    # Open the CSV file for writing
    with open(writefilename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        # Column 1:	Vehicle ID
        # Column 2:	Frame ID
        # Column 3:	Lane ID
        # Column 4:	LocalY
        # Column 5:	Mean Speed
        # Column 6:	Mean Acceleration
        # Column 7:	Vehicle length
        # Column 8:	Vehicle Class ID
        # Column 9:	Follower ID
        # Column 10: Leader ID

        writer.writerow(["VehicleID", "Time", "LaneID", "LocalY", "MeanSpeed", "MeanAccel", "VehLength", "VehClass", "FollowerID", "LeaderID"])
        # vehicle_id = "carflow1.131"
        # Run simulation steps
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            # Get simulation time
            simulation_time = traci.simulation.getTime()

            # Get IDs of all vehicles
            vehicle_ids = traci.vehicle.getIDList()

            # Iterate over all vehicles
            for vehicle_id in vehicle_ids:
                # Get vehicle position and speed
                position = traci.vehicle.getPosition(vehicle_id)
                laneid = traci.vehicle.getLaneID(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                accel = traci.vehicle.getAcceleration(vehicle_id)
                cls = traci.vehicle.getVehicleClass(vehicle_id)

                try:
                    lead_id, lead_dist = traci.vehicle.getLeader(vehicle_id)
                except TypeError:
                    # no leader
                    lead_id = -1
                foll_id, foll_dist = traci.vehicle.getFollower(vehicle_id)
                length = traci.vehicle.getLength(vehicle_id)

                # Write data to the CSV file - similar to NGSIM schema
                # csv_file.write(f"{vehicle_id} {simulation_time} {laneid} {position[0]} {speed} {accel} {length} {cls} {foll_id} {lead_id}\n")
                writer.writerow([vehicle_id, simulation_time, laneid, position[0], speed, accel, length, cls, foll_id, lead_id])

            # try to overwite acceleration of one vehicle
            # if 300< step <400:
            #     traci.vehicle.setSpeed(vehicle_id, 0)
            # Simulate one step
            traci.simulationStep()
            step += 1

    # Close connection
    traci.close()
    print("Complete!")

    return

def reorder_by_id(trajectory_file, link_names="all", lane_name=""):
    # read the trajectory file to see if it is ordered by time (time increases from previous row)
    # assumption: trajectory_file is either ordered by time or by vehicleID
    # separate files by laneID if bylane is True
    
    prev_time = -1
    ordered_by_time = True
    with open(trajectory_file, mode='r') as file:
        csv_reader = csv.reader(file)
        _ = next(csv_reader)
        for columns in csv_reader:
            # Split the line into columns
            # columns = line.strip().split()
            # columns = line[0].strip().split()
            # print(columns)
            try:
                curr_time = float(columns[1]) # int(columns[0])
            except IndexError:
                columns = columns[0].split(',')
                # print(columns)
                curr_time = float(columns[1]) 
                
            if curr_time < prev_time:
                print(trajectory_file + " is NOT ordered by time")
                ordered_by_time = False
                return
            prev_time = curr_time
    
    # reorder this file by ID
    if ordered_by_time:
        print(trajectory_file + " is ordered by time, reordering it by vehicleID...")
        with open(trajectory_file, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            # print(csv_reader)
            # rows = [row[0].strip().split() for row in csv_reader]
            rows = [row for row in csv_reader]
        # Sort the rows by vehicleID and then by time within each vehicleID
        print("Start time: ", rows[0][1])
        print("End time: ", rows[-1][1])
        print("Start pos: ", rows[0][3]) # TODO may not be accurate
        print("End pos: ", rows[-1][3])

        rows.sort(key=lambda x: (x[0], float(x[1])))

        # if bylane==True:
        #     # Organize rows by laneID
        #     lanes = defaultdict(list)
        #     for row in rows:
        #         lane_id = row[2]  # assuming laneID is in the third column
        #         lanes[lane_id].append(row)
            
        #     for lane_id, lane_rows in lanes.items():
        #         output_file = trajectory_file.replace(".csv", f"_{lane_id}.csv")
        #         with open(output_file, mode='w') as file:
        #             file.write(",".join(headers) + "\n")  # Write the headers
        #             for row in lane_rows:
        #                 file.write(" ".join(str(num) for num in row) + "\n")
        if link_names == "all":
            # Write the sorted rows to a new CSV file
            output_file = trajectory_file.replace(".csv", "_byid.csv")
            with open(output_file, mode='w', newline="") as file:
                csv_writer = csv.writer(file)
                file.write(",".join(str(num) for num in headers)+"\n")  # Write the headers
                # csv_writer.writerows(rows)
                for row in rows:
                    # file.write(" ".join(str(num) for num in row)+"\n")
                    csv_writer.writerow(row)

        else: # write selected link names
            # Write the sorted rows to a new CSV file
            output_file = trajectory_file.replace(".csv", f"_{lane_name}.csv")
            with open(output_file, mode='w') as file:
                # csv_writer = csv.writer(file)
                file.write(",".join(str(num) for num in headers)+"\n")  # Write the headers
                # csv_writer.writerows(rows)
                for row in rows:
                    lane_id = row[2]
                    if lane_id in link_names:
                        file.write(" ".join(str(num) for num in row)+"\n")

    return

if __name__ == "__main__":
    # Uncomment and use one of the functions based on your needs
    # script_path = r"C:\Program Files (x86)\Eclipse\Sumo\tools\createVehTypeDistribution.py"
    # if not os.path.isfile(script_path):
    #     print(f"Error: {script_path} does not exist.")
    #     sys.exit(1)
    
    # # Execute the Python script
    # vehDistFile = "vTypeDistributions.add.xml"
    # if os.path.exists(vehDistFile):
    #     # If the file exists, delete it
    #     os.remove(vehDistFile)
    #     print("removed ",vehDistFile)
    # subprocess.run(["python", script_path] + ["vTypeDist.txt"]) # add aguments

    # ============== simulate & save data ==================
    # fcd_name = "fcd_onramp_cf_rho"
    # lane_id = 2 # left-most lane is lane 1
    # if lane_id == 1:
    #     link_names = ["E0_1", "E1_1", "E2_2", "E4_1"]
    # elif lane_id == 2:
    #     link_names = ["E0_0", "E1_0", "E2_1", "E4_0"]
    # elif lane_id == "onramp":
    #     link_names = ["ramp_0", "E2_0"]
    # macro.reorder_by_id(fcd_name+".csv", link_names=link_names, lane_name="lane2")
    # macro_data = macro.compute_macro(fcd_name+"_lane2.csv", dx=10, dt=10, start_time=0, end_time=480, start_pos =0, end_pos=1300,
    #                                  save=True, plot=True)
    # macro_data = macro.compute_macro("onramp_trajectories_ramp_0.csv", dx=10, dt=10, save=True, plot=False)

    # macro_pkl = r'Z:\VMS_Data\I24MOTION\SUMO\macro_data\macro_onramp_trajectories_E4_1.pkl'
    # with open(macro_pkl, 'rb') as file:
    #     macro_data = pickle.load(file)
    # print(macro_data["speed"])
    # macro.plot_macro(macro_data)


    # ================== generate some NGSIM-like data==================
    # onramp.run_sumo(sim_config="onramp_gt.sumocfg", fcd_output ="trajs_gt.xml")
    # write_vehicle_trajectories_to_csv(readfilename="onramp_gt", writefilename="trajs_gt.csv")
    reorder_by_id(trajectory_file="trajs_gt.csv")