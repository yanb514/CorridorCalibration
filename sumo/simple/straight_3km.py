import traci
import traci.constants as tc
import xml.etree.ElementTree as ET
import os
import sys
import subprocess
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
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
    with open(writefilename, 'w') as csv_file:
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

        csv_file.write("VehicleID, Time, LaneID, LocalY, MeanSpeed, MeanAccel, VehLength, VehClass, FollowerID, LeaderID\n")
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
                speed = traci.vehicle.getSpeed(vehicle_id)
                accel = traci.vehicle.getAcceleration(vehicle_id)
                cls = traci.vehicle.getVehicleClass(vehicle_id)

                # Write data to the CSV file - similar to NGSIM schema
                csv_file.write(f"{vehicle_id} {simulation_time} {-1} {position[0]} {speed} {accel} {-1} {cls} {-1} {-1}\n")

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

    sumo_name = 'straight_3km'
    file_name = sumo_name + "_trajectories.csv"
    # write_vehicle_trajectories_to_csv(sumo_name, file_name)
    
    vis.scatter_time_space("", file_name, highlight_leaders=False)

