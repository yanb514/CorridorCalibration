import os
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
import optuna
import subprocess
import xml.etree.ElementTree as ET
import shutil
import logging
from datetime import datetime
import sys
import glob
import json
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
optuna.logging.set_verbosity(optuna.logging.ERROR)


# ================ Configuration ====================
SCENARIO = "onramp"
N_TRIALS = 10000
SUMO_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script directory
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)
measurement_locations = ['upstream_0', 'upstream_1', 'merge_0', 'merge_1', 'merge_2', 'downstream_0', 'downstream_1']

# Define parameter ranges for calibration
param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
min_val = [30.0, 1.0, 1.0, 1.0, 0.5]
max_val = [35.0, 3.0, 4.0, 3.0, 2.0]

# ================ Discriminator Model ====================
# class Discriminator(nn.Module):
#     def __init__(self, traffic_shape, od_shape):
#         """
#         Discriminator model for cGAN.
        
#         Args:
#             traffic_shape (tuple): Shape of spatiotemporal traffic patterns (e.g., T × L × F).
#             od_shape (tuple): Shape of time-varying OD matrices (e.g., routes × demand × time).
#         """
#         super(Discriminator, self).__init__()
        
#         # Process traffic patterns with 3D convolutional layers
#         self.traffic_conv = nn.Sequential(
#             nn.Conv3d(in_channels=traffic_shape[0], out_channels=32, kernel_size=(3, 3, 3), padding=1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool3d(kernel_size=(2, 2, 2)),
#             nn.Flatten()
#         )
        
#         # Process OD matrices with 1D convolutional layers (over time)
#         self.od_conv = nn.Sequential(
#             nn.Conv1d(in_channels=od_shape[0], out_channels=32, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Flatten()
#         )
        
#         # Fully connected layers
#         self.fc = nn.Sequential(
#             nn.Linear(32 * (traffic_shape[1] // 2) * (traffic_shape[2] // 2) * (traffic_shape[3] // 2) + 32 * (od_shape[2] // 2), 64),
#             nn.LeakyReLU(0.2),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, traffic_pattern, od_matrix):
#         # Process traffic patterns
#         traffic_features = self.traffic_conv(traffic_pattern)
        
#         # Process OD matrices
#         od_features = self.od_conv(od_matrix)
        
#         # Concatenate features
#         combined = torch.cat([traffic_features, od_features], dim=1)
        
#         # Final output
#         output = self.fc(combined)
#         return output

# ================ SUMO Simulation Functions ====================
def run_sumo(sim_config):
    """Run a SUMO simulation with the given configuration."""
    command = ['sumo', '-c', sim_config, '--no-step-log', '--xml-validation', 'never']
    subprocess.run(command, check=True)

def create_temp_config(param, trial_number):
    """Create temporary SUMO configuration files for each trial."""
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # Update .rou.xml file with new parameters
    rou_tree = ET.parse(f"{SCENARIO}.rou.xml")
    rou_root = rou_tree.getroot()
    for vtype in rou_root.findall('vType'):
        if vtype.get('id') == 'trial':
            for key, val in param.items():
                vtype.set(key, str(val))
            break
    new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
    rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)
    
    # Copy other necessary files
    shutil.copy(f"{SCENARIO}.net.xml", os.path.join(output_dir, f"{trial_number}_{SCENARIO}.net.xml"))
    shutil.copy('detectors.add.xml', os.path.join(output_dir, f"{trial_number}_detectors.add.xml"))
    
    # Update .sumocfg file
    sumocfg_tree = ET.parse(f"{SCENARIO}.sumocfg")
    sumocfg_root = sumocfg_tree.getroot()
    input_element = sumocfg_root.find('input')
    if input_element is not None:
        input_element.find('route-files').set('value', f"{trial_number}_{SCENARIO}.rou.xml")
        input_element.find('net-file').set('value', f"{trial_number}_{SCENARIO}.net.xml")
        input_element.find('additional-files').set('value', f"{trial_number}_detectors.add.xml")
    new_sumocfg_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.sumocfg")
    sumocfg_tree.write(new_sumocfg_file_path, encoding='UTF-8', xml_declaration=True)
    
    return new_sumocfg_file_path, output_dir

def extract_od_matrix(temp_path):
    """
    Extract time-varying OD matrix from the .rou file.
    
    Returns:
        od_matrix (np.ndarray): A 2D array of shape (routes × time).
    """
    rou_file_path = glob.glob(os.path.join(temp_path, "*.rou.xml"))[0]
    tree = ET.parse(rou_file_path)
    root = tree.getroot()
    
    # Initialize OD matrix (example: 3 routes, 5 time steps)
    num_routes = config[SCENARIO]["N_ROUTES"]
    num_intervals = config[SCENARIO]["N_INTERVALS"]
    demand_matrix = np.zeros((num_routes, num_intervals))
    
    # Extract unique routes from route definitions
    routes = [route.get("id") for route in root.findall("route")]
    route_index = {route: i for i, route in enumerate(routes)}

    # Extract flow data
    for flow in root.findall("flow"):
        route = flow.get("route")
        vehs_per_hour = float(flow.get("vehsPerHour"))
        begin = int(flow.get("begin"))
        interval_idx = begin // 1800  # Assuming time intervals of 1800s

        if route in route_index:
            demand_matrix[route_index[route], interval_idx] = vehs_per_hour
    
    return demand_matrix

# ================ Objective Function ====================
def objective(trial):
    """Objective function for optimization with cGAN."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    } # TODO: add constraints using stability and RDC
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param, trial.number)

    # Run SUMO simulation
    run_sumo(temp_config_path)
    
    # Extract simulated traffic patterns (flow, density, speed)
    simulated_output = reader.extract_sim_meas(["trial_"+ location for location in measurement_locations],
                                        file_dir = temp_path)
    
    # Extract time-varying OD matrix from the .rou file
    od_matrix = extract_od_matrix(temp_path)
    
    # Reshape simulated output and OD matrix for the discriminator
    traffic_pattern = np.stack([simulated_output['flow'], simulated_output['density'], simulated_output['speed']], axis=-1)
    traffic_pattern = np.expand_dims(traffic_pattern, axis=0)  # Add batch dimension
    traffic_pattern = torch.tensor(traffic_pattern, dtype=torch.float32)
    
    # od_matrix = np.expand_dims(od_matrix, axis=0)  # Add batch dimension
    od_matrix = torch.tensor(od_matrix, dtype=torch.float32)
    
    # Evaluate realism using the discriminator
    realism_score = discriminator(traffic_pattern, od_matrix).item()
    
    # Adversarial loss: encourage the discriminator to classify the simulated pattern as real
    adversarial_loss = -torch.log(torch.tensor(realism_score))
    
    # Train the discriminator with real and simulated data
    train_discriminator(discriminator, real_traffic_patterns, real_od_matrices, traffic_pattern, od_matrix)
    
    # Clear temporary files
    clear_directory(os.path.join("temp", str(trial.number)))
    
    return adversarial_loss.item()

# ================ Discriminator Training Function ====================
def train_discriminator(discriminator, real_traffic_patterns, real_od_matrices, simulated_traffic_patterns, simulated_od_matrices):
    """
    Train the discriminator on real and simulated traffic patterns.
    """
    # Combine real and simulated data
    X_traffic = torch.cat([real_traffic_patterns, simulated_traffic_patterns], dim=0)
    X_od = torch.cat([real_od_matrices, simulated_od_matrices], dim=0)
    y = torch.cat([torch.ones(len(real_traffic_patterns)), torch.zeros(len(simulated_traffic_patterns))], dim=0)
    
    # Shuffle the data
    indices = torch.randperm(len(X_traffic))
    X_traffic = X_traffic[indices]
    X_od = X_od[indices]
    y = y[indices]
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Train the discriminator
    optimizer.zero_grad()
    outputs = discriminator(X_traffic, X_od)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# ================ Main Script ====================
if __name__ == "__main__":
    # Initialize logging
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_dir = '_log'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_file = os.path.join(log_dir, f'{current_time}_optuna_log_{EXP}.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # # Initialize discriminator
    # traffic_shape = (3, 10, 10, 10)  # Example shape: (channels × time × space × features)
    # od_shape = (3, 5)  # Example shape: (routes × time)
    # discriminator = Discriminator(traffic_shape, od_shape)

    # # Load real-world data for discriminator training
    # real_traffic_patterns = torch.tensor(np.random.rand(100, *traffic_shape), dtype=torch.float32)  # Replace with real data
    # real_od_matrices = torch.tensor(np.random.rand(100, *od_shape), dtype=torch.float32)  # Replace with real data

    # # Run optimization
    # sampler = optuna.samplers.TPESampler(seed=10)
    # study = optuna.create_study(direction='minimize', sampler=sampler)
    # study.optimize(objective, n_trials=N_TRIALS, n_jobs=os.cpu_count()-1)

    # # Save results
    # print('Best parameters:', study.best_params)

