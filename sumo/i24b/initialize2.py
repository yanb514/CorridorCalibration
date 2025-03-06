'''
generate flows in .rou.xml by detector measurements
quick and easy, not precise
'''
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict

# Configuration - Update these paths to match your files
DETECTOR_LOCATIONS_FILE = "12-15_lanelocs.xml"
ROUTES_FILE = "i24b.rou.xml"
DETECTOR_MEASUREMENTS_FILE = "detector_measurements_i24wRDSdetectors.csv"
OUTPUT_ROUTES_FILE = "i24_with_flows.rou.xml"
START_MINUTE = 380  # Start of measurement window (minutes)
END_MINUTE = 390    # End of measurement window (minutes)

def main():
    # Convert minutes to seconds for SUMO
    start_time = START_MINUTE * 60
    end_time = END_MINUTE * 60

    # Step 1: Parse detector locations
    detector_edges = {}
    det_tree = ET.parse(DETECTOR_LOCATIONS_FILE)
    for poi in det_tree.findall('poi'):
        detector_id = poi.get('id')
        lane = poi.get('lane')
        edge = lane.split('_')[0]
        detector_edges[detector_id] = edge

    # Step 2: Parse routes
    routes = {}
    rou_tree = ET.parse(ROUTES_FILE)
    for route in rou_tree.findall('route'):
        route_id = route.get('id')
        edges = route.get('edges').split()
        routes[route_id] = edges

    # Step 3: Initialize data structure for time-varying flows
    time_interval_data = defaultdict(
        lambda: defaultdict(lambda: {'total_q': 0, 'total_v': 0.0, 'count': 0})
    )

    # Step 4: Process detector measurements
    with open(DETECTOR_MEASUREMENTS_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                time_min = float(row['Time'])
                interval = int(time_min)  # 1-minute interval
            except ValueError:
                continue
            
            # Filter data within our time window
            if not (START_MINUTE <= interval < END_MINUTE):
                continue

            detector_id = row['Detector']
            edge = detector_edges.get(detector_id)
            if not edge:
                continue

            # Find routes containing this edge
            for route_id, edges in routes.items():
                if edge in edges:
                    q = int(row['qPKW'])
                    v = float(row['vPKW']) # TODO v=0 if no vehicle
                    
                    # Aggregate data per route per time interval
                    data = time_interval_data[interval][route_id]
                    data['total_q'] += q
                    data['total_v'] += v
                    data['count'] += 1

    # Step 5: Create flow elements
    root = rou_tree.getroot()
    
    # Create time-varying flows for each interval and route
    for interval in range(START_MINUTE, END_MINUTE):
        interval_data = time_interval_data.get(interval, {})
        
        for route_id, data in interval_data.items():
            if data['count'] == 0:
                continue
            
            avg_speed = data['total_v'] / data['count']
            total_q = data['total_q']
            
            # Convert minute interval to seconds
            begin = interval * 60
            end = (interval + 1) * 60
            
            # Create flow element
            flow_elem = ET.Element('flow', {
                'id': f'flow_{route_id}_{interval}',
                'route': route_id,
                'begin': str(begin-START_MINUTE*60),
                'end': str(end-START_MINUTE*60),
                'vehsPerHour': str(total_q * 60),  # Convert minute data to hourly rate
                'speed': f"{avg_speed:.2f}",
                'type': 'hdv'
            })
            root.append(flow_elem)

    # Step 6: Save modified route file
    rou_tree.write(OUTPUT_ROUTES_FILE, encoding='UTF-8', xml_declaration=True)
    print(f"Generated time-varying route file saved to {OUTPUT_ROUTES_FILE}")

if __name__ == '__main__':
    main()