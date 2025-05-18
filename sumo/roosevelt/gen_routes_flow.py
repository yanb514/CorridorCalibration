"""
Automates the complete route generation and flow assignment process for SUMO networks
This script implements the following steps:
1. Generate all possible routes given a network file "XXX.net.xml", origin edges and destination edges
    generate_all_routes()
2. Generate random flows on the routes from step 1
    add_random_flows()  
3. Clean up network
    remove_disallow_attributes()
"""
import sys
import subprocess
import sumolib
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import random

# ================= Configuration =================
NETWORK_FILE = "roosevelt.net.xml"
OUTPUT_ROUTES_FILE = "roosevelt.rou.xml"
FIND_ALL_ROUTES_SCRIPT = "/home/ywan1649/Documents/sumo/tools/findAllRoutes.py"
# $SUMO_HOME/tools/findAllRoutes.py

# Flow generation parameters
SIMULATION_DURATION = 600  # in seconds
MIN_FLOW = 10              # Minimum vehicles per hour
MAX_FLOW = 20             # Maximum vehicles per hour
VEHICLE_TYPE = "hdv"        # SUMO vehicle type
SELECTED_SOURCE = ["435561321#0", "435381830#0"] # to select source and target edges on Roosevelt E-W
SELECTED_TARGET = ["1000784669#1", "314011259#1"]

# =============== Core Functions ================
def find_source_target_edges():
    """Identify source and target edges in the network"""
    try:
        net = sumolib.net.readNet(NETWORK_FILE)
    except FileNotFoundError:
        sys.exit(f"Error: Network file {NETWORK_FILE} not found")

    # Find source edges (no incoming connections)
    source_edges = {e.getID() for e in net.getEdges() if not e.getIncoming()}
    
    # Find target edges (no outgoing connections)
    target_edges = {e.getID() for e in net.getEdges() if not e.getOutgoing()}

    if not source_edges:
        sys.exit("Error: No source edges found in the network")
    if not target_edges:
        sys.exit("Error: No target edges found in the network")

    return source_edges, target_edges

def generate_all_routes(source_edges=None, target_edges=None):
    """Generate all possible routes using SUMO's findAllRoutes.py
        source_edges: a list of source edges
        target_edges: a list of target edges
        If source_edges and target_edges are None, then generate routes from all possible combinations 
        of sources and targets using find_source_target_edges().
        CAUTION: if a network is large, this will generate a large number of OD routes
    """
    if source_edges is None and target_edges is None:
        print("\n Source and target edges not provided. Identifying source and target edges...")
        source_edges, target_edges = find_source_target_edges()
        
    else:
        source_edges = SELECTED_SOURCE
        target_edges = SELECTED_TARGET
        
    print(f"   Found {len(source_edges)} source edges")
    print(f"   Found {len(target_edges)} target edges")

    source_str = ",".join(source_edges)
    target_str = ",".join(target_edges)
    # print(source_str, target_str)

    command = [
        "python",
        FIND_ALL_ROUTES_SCRIPT,
        "-n", NETWORK_FILE,
        "-o", OUTPUT_ROUTES_FILE,
        "-s", source_str,
        "-t", target_str
    ]

    try:
        subprocess.run(command, check=True)
        print("\nSuccessfully generated all routes")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error generating routes: {str(e)}")
    except FileNotFoundError:
        sys.exit(f"findAllRoutes.py not found at {FIND_ALL_ROUTES_SCRIPT}")

def add_random_flows():
    """Add random traffic flows to generated routes"""
    try:
        rou_tree = ET.parse(OUTPUT_ROUTES_FILE)
    except FileNotFoundError:
        sys.exit(f"Route file {OUTPUT_ROUTES_FILE} not found")

    root = rou_tree.getroot()

    # Remove existing flows
    for elem in list(root):
        if elem.tag == 'flow':
            root.remove(elem)

    # Add vehicle type definitions at the beginning
    vtypes = [
        ET.Element('vType', {
            'id': 'hdv',
            'length': '4.3',
            'carFollowModel': 'IDM',
            'emergencyDecel': '4.0',
            'laneChangeModel': 'SL2015',
            'latAlignment': 'arbitrary',
            'lcKeepRight': '0.0',
            'lcOvertakeRight': '0.0',
            'maxSpeed': '20.55',
            'minGap': '2.5',
            'accel': '1.0',
            'decel': '2',
            'tau': '1.4',
            'lcSublane': '1.0',
            'maxSpeedLat': '5',
            'lcAccelLat': '0.7',
            'minGapLat': '0.4',
            'lcStrategic': '10',
            'lcCooperative': '1.0',
            'lcPushy': '0.4',
            'lcImpatience': '0.',
            'lcSpeedGain': '1.5'
        }),
        ET.Element('vType', {
            'id': 'cav',
            'length': '4.3',
            'carFollowModel': 'IDM',
            'emergencyDecel': '4.0',
            'laneChangeModel': 'SL2015',
            'latAlignment': 'arbitrary',
            'lcKeepRight': '0.0',
            'lcOvertakeRight': '0.0',
            'maxSpeed': '20.55',
            'minGap': '2.5',
            'accel': '1.0',
            'decel': '2',
            'tau': '1.4',
            'lcSublane': '1.0',
            'maxSpeedLat': '5',
            'lcAccelLat': '1.2',
            'minGapLat': '0.4',
            'lcStrategic': '50',
            'lcCooperative': '1.0',
            'lcPushy': '0.4',
            'lcImpatience': '0.9',
            'lcSpeedGain': '1.5'
        })
    ]

    # Insert vTypes as the first elements
    for vtype in reversed(vtypes):
        root.insert(0, vtype)

    # Add random flows
    for route in root.findall('route'):
        route_id = route.get('id')
        route.set('color', f"{random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)}")
        
        flow_elem = ET.Element('flow', {
            'id': f"flow_{route_id}",
            'route': route_id,
            'begin': "0",
            'end': str(SIMULATION_DURATION),
            'vehsPerHour': str(random.randint(MIN_FLOW, MAX_FLOW)),
            'departSpeed': "desired",
            'type': VEHICLE_TYPE
        })
        root.append(flow_elem)

    # Save modified file
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(OUTPUT_ROUTES_FILE, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    print(f"\nAdded random flows to {OUTPUT_ROUTES_FILE}")


def remove_disallow_attributes(input_file, output_file):
    """
    Remove all 'disallow' attributes from a SUMO network XML file
    while maintaining proper XML formatting and structure.
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Remove 'disallow' attributes from all elements
    for elem in root.iter():
        if 'disallow' in elem.attrib:
            del elem.attrib['disallow']

    # Convert to string for pretty formatting
    rough_xml = ET.tostring(root, encoding='utf-8')
    parsed_xml = minidom.parseString(rough_xml)

    # Preserve XML declaration with original attributes
    pretty_xml = parsed_xml.toprettyxml(indent="    ", encoding='utf-8').decode('utf-8')
    
    # Remove extra newlines added by minidom
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


# =============== Main Execution ================
if __name__ == "__main__":
    print("=== Starting Automated Route Generation ===")
    
    # Step 1: Generate all possible routes
    print("\n1. Generating all possible routes...")
    # generate_all_routes(source_edges=SELECTED_SOURCE, target_edges=SELECTED_TARGET) # generate all routes from the given origins and destination edges
    generate_all_routes() # generate all feasible routes from the entire network

    # Step 2: Add random flows
    print("\n2. Adding random traffic flows...")
    add_random_flows()

    # Step 3: clean up network file (remove "disallow" attribute)
    print("\n3. Cleaning up network file...")
    # Usage example
    remove_disallow_attributes(
        NETWORK_FILE,
        NETWORK_FILE
    )

    print("\n=== Process completed successfully ===")
    print(f"Final output file: {OUTPUT_ROUTES_FILE}")