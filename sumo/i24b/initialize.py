import pandas as pd
import sumolib
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

# Configuration
NET_FILE = "i24b.net.xml"
DETECTOR_FILE = "12-15_detectors.xml"
DETECTOR_DATA_CSV = "detector_measurements_i24wRDSdetectors.csv"
OUTPUT_PREFIX = "calibrated"
# SUMO_HOME = os.environ.get("SUMO_HOME", "/home/ywan1649/Documents/sumo")  # Update this
SUMO_HOME = "/packages/apps/spack/21/opt/spack/linux-rocky8-zen3/gcc-12.1.0/sumo-1.19.0-hkbmitts4svgguaerh3osctddszoeu4m"
SUMO_TOOLS = "/home/ywan1649/Documents/sumo/tools"
# SUMO_HOME = ≈

print(SUMO_HOME)

# Step 1: Map detectors to edges using ElementTree
def map_detectors_to_edges():
    net = sumolib.net.readNet(NET_FILE)
    tree = ET.parse(DETECTOR_FILE)
    root = tree.getroot()
    
    detector_map = {}
    for det in root.findall('inductionLoop'):
        det_id = det.get('id')
        lane_id = det.get('lane')
        edge_id = net.getLane(lane_id).getEdge().getID()
        detector_map[det_id] = edge_id
    
    return detector_map

# Step 2: Process detector data and aggregate to edge-level counts
def process_detector_data(detector_map):
    df = pd.read_csv(DETECTOR_DATA_CSV, sep=';')
    df['Edge'] = df['Detector'].map(detector_map)
    edge_counts = df.groupby(['Edge', 'Time']).agg({'qPKW': 'sum'}).reset_index()
    edge_counts.to_csv(f"{OUTPUT_PREFIX}_edge_counts.csv", index=False)
    return edge_counts


# Step 3: Run flowrouter.py from SUMO tools
def run_flowrouter():
    cmd = [
        "python", 
        os.path.join(SUMO_TOOLS, "detector", "flowrouter.py"),
        "--net-file", NET_FILE,                   # Mandatory: Network file
        "--detector-file", DETECTOR_FILE,         # Mandatory: Detector definitions (XML)
        "--detector-flow-files", DETECTOR_DATA_CSV,  # Mandatory: Flow data (XML)
        "-o", f"{OUTPUT_PREFIX}_od_matrix.csv",  # Corrected argument
    ]
    subprocess.run(cmd, check=True)

# Step 4: Generate routes from OD matrix
def generate_routes():
    # Convert OD matrix to trips
    subprocess.run([
        os.path.join(SUMO_HOME, "bin", "od2trips"),
        "-n", NET_FILE,
        "-d", f"{OUTPUT_PREFIX}_od_matrix.csv",
        "-o", f"{OUTPUT_PREFIX}_trips.trips.xml"
    ], check=True)

    # Generate routes with duarouter
    subprocess.run([
        os.path.join(SUMO_HOME, "bin", "duarouter"),
        "-n", NET_FILE,
        "-t", f"{OUTPUT_PREFIX}_trips.trips.xml",
        "-o", f"{OUTPUT_PREFIX}_routes.rou.xml",
        "--ignore-errors",
        "--remove-loops"
    ], check=True)

# Step 5: Add speed information using ElementTree
def add_speed_information():
    tree = ET.parse(f"{OUTPUT_PREFIX}_routes.rou.xml")
    root = tree.getroot()
    
    # Create vehicle type element
    vtype = ET.Element("vType", {
        "id": "calibrated_type",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "speedFactor": "1.0",
        "speedDev": "0.1"
    })
    
    # Insert at beginning of XML
    root.insert(0, vtype)
    
    # Add type to all vehicles
    for vehicle in root.findall("vehicle"):
        vehicle.set("type", "calibrated_type")
    
    # Write back with XML declaration
    tree.write(f"{OUTPUT_PREFIX}_routes.rou.xml", encoding="UTF-8", xml_declaration=True)

# Main workflow
def main():
    print("Step 1/4: Mapping detectors to edges...")
    detector_map = map_detectors_to_edges()
    
    print("Step 2/4: Processing detector data...")
    process_detector_data(detector_map)
    
    print("Step 3/4: Running flowrouter...")
    run_flowrouter()
    
    print("Step 4/4: Generating routes...")
    generate_routes()
    
    # print("Optional: Adding speed information...")
    # add_speed_information()
    
    print(f"Done! Calibrated routes saved to {OUTPUT_PREFIX}_routes.rou.xml")

if __name__ == "__main__":
    main()