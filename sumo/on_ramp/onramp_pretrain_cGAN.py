"""
1. Obtain training data
    a. obtain "real" detector ouputs (.out.xml)
        a. Fixed parameters, mildly varying demands
    b. obtain "fake" detector outputs
        a. Varying parameters, mildly varying demands
2. save data in
    /data/SCENARIO/real
    /data/SCENARio/sim
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import xml.etree.ElementTree as ET
import shutil
import logging
from datetime import datetime
import sys
import glob
import json
import random
import multiprocessing
from functools import partial

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
from onramp_calibrate import clear_directory
# optuna.logging.set_verbosity(optuna.logging.ERROR)


# ================ Configuration ====================
SCENARIO = "onramp"
N_TRIALS = 10000
SUMO_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script directory
FLOW_STD = 5
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)
measurement_locations = ['upstream_0', 'upstream_1', 'merge_0', 'merge_1', 'merge_2', 'downstream_0', 'downstream_1']

# Define parameter ranges for calibration
param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
min_val = [30.0, 1.0, 1.0, 1.0, 0.5]
max_val = [35.0, 3.0, 4.0, 3.0, 2.0]

# ================ Discriminator Model ====================

def simulate_real_worker(i, measurement_locations):
    """
    Runs SUMO for a single sample and extracts traffic patterns.
    """
    temp_config_path, temp_path = create_temp_config(param=None, trial_number=i, type="real")
    run_sumo(temp_config_path)

    # Extract simulated traffic volumes
    real_meas = reader.extract_sim_meas(
        ["trial_" + location for location in measurement_locations],
        file_dir=temp_path
    )

    # 'real_meas' is a dictionary with keys "flow", "density", "speed", each holding a 2D array (time x space)
    flow = real_meas["flow"]
    density = real_meas["density"]
    speed = real_meas["speed"]

    # Stack the features (flow, density, speed) along a new dimension to form (time, space, features)
    traffic = np.stack([flow, density, speed], axis=-1)  # Shape: (time, space, features)
    demand = extract_od_matrix(temp_path)

    return traffic, demand


def generate_real_data(num_samples):
    """
    Generate real traffic data samples using SUMO in parallel.
    """

    # Use multiprocessing to parallelize the SUMO runs
    with multiprocessing.Pool(processes=min(num_samples, multiprocessing.cpu_count())) as pool:
        results = pool.map(partial(simulate_real_worker, measurement_locations=measurement_locations), range(num_samples))

    # Separate traffic and demand for saving
    traffic_patterns = [result[0] for result in results]  # Extract traffic data
    demand_data = [result[1] for result in results]      # Extract demand data

    # Convert the list of traffic patterns to a tensor
    traffic_patterns_tensor = torch.tensor(np.array(traffic_patterns), dtype=torch.float32)
    print(f"Traffic patterns tensor shape: {traffic_patterns_tensor.shape}")

    # Convert the list of demand data to a tensor
    demand_tensor = torch.tensor(np.array(demand_data), dtype=torch.float32)
    print(f"Demand tensor shape: {demand_tensor.shape}")

    # Clear temporary directory
    clear_directory("temp")

    # Save both traffic_patterns_tensor and demand_tensor to "/data/SCENARIO/sim"
    save_traffic_path = os.path.abspath(f"../../data/{SCENARIO}/real/traffic_patterns.pt")
    save_demand_path = os.path.abspath(f"../../data/{SCENARIO}/real/demand.pt")
    
    os.makedirs(os.path.dirname(save_traffic_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_demand_path), exist_ok=True)
    
    torch.save(traffic_patterns_tensor, save_traffic_path)
    torch.save(demand_tensor, save_demand_path)
    
    print(f"Saved traffic patterns tensor to {save_traffic_path}")
    print(f"Saved demand tensor to {save_demand_path}")

    return 


def simulate_sim_worker(i, measurement_locations):
    """
    Runs SUMO for a single sample and extracts traffic patterns and demand.
    """
    # Generate random driver parameters
    driver_param = {
        param_name: random.uniform(min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    }

    # Create temp configuration for the simulation
    temp_config_path, temp_path = create_temp_config(param=driver_param, trial_number=i, type="sim")
    
    # Run SUMO simulation
    run_sumo(temp_config_path)

    # Extract simulated traffic volumes
    real_meas = reader.extract_sim_meas(
        ["trial_" + location for location in measurement_locations],
        file_dir=temp_path
    )

    # 'real_meas' is a dictionary with keys "flow", "density", "speed", each holding a 2D array (time x space)
    flow = real_meas["flow"]
    density = real_meas["density"]
    speed = real_meas["speed"]

    # Stack the features (flow, density, speed) along a new dimension to form (time, space, features)
    traffic = np.stack([flow, density, speed], axis=-1)  # Shape: (time, space, features)
    
    # Extract OD matrix (demand)
    demand = extract_od_matrix(temp_path)

    return traffic, demand

def generate_sim_data(num_samples):
    """
    Generate real traffic data samples using SUMO in parallel.
    """

    # Use multiprocessing to parallelize the SUMO runs
    with multiprocessing.Pool(processes=min(num_samples, multiprocessing.cpu_count())) as pool:
        # For each sample, simulate traffic and demand
        results = pool.map(partial(simulate_sim_worker, measurement_locations=measurement_locations), range(num_samples))

    # Separate traffic and demand for saving
    traffic_patterns = [result[0] for result in results]  # Extract traffic data
    demand_data = [result[1] for result in results]      # Extract demand data

    # Convert the list of traffic patterns to a tensor
    traffic_patterns_tensor = torch.tensor(np.array(traffic_patterns), dtype=torch.float32)
    
    # Print shape to confirm
    print(f"Traffic patterns tensor shape: {traffic_patterns_tensor.shape}")

    # Convert the list of demand data to a tensor
    demand_tensor = torch.tensor(np.array(demand_data), dtype=torch.float32)
    
    # Print shape to confirm
    print(f"Demand tensor shape: {demand_tensor.shape}")

    # Clear temporary directory
    clear_directory("temp")

    # Save both traffic_patterns_tensor and demand_tensor to "/data/SCENARIO/sim"
    save_traffic_path = os.path.abspath(f"../../data/{SCENARIO}/sim/traffic_patterns.pt")
    save_demand_path = os.path.abspath(f"../../data/{SCENARIO}/sim/demand.pt")
    
    print(f"Saving to: {save_traffic_path} and {save_demand_path}")
    os.makedirs(os.path.dirname(save_traffic_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_demand_path), exist_ok=True)
    
    torch.save(traffic_patterns_tensor, save_traffic_path)
    torch.save(demand_tensor, save_demand_path)
    
    print(f"Saved traffic patterns tensor to {save_traffic_path}")
    print(f"Saved demand tensor to {save_demand_path}")

    return traffic_patterns_tensor, demand_tensor

# ================ SUMO Simulation Functions ====================
def run_sumo(sim_config):
    """Run a SUMO simulation with the given configuration."""
    command = ['sumo', '-c', sim_config, '--no-step-log', '--xml-validation', 'never']
    subprocess.run(command, check=True)

def create_temp_config(param, trial_number, type="real"):
    """Create temporary SUMO configuration files for each trial."""
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # Update .rou.xml file with new parameters
    if type == "real": # do not change parameters, add noise on flw
        # shutil.copy(f"{SCENARIO}_gt.rou.xml", os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml"))
        rou_tree = ET.parse(f"{SCENARIO}_gt.rou.xml")
        rou_root = rou_tree.getroot()
        for flow in rou_root.findall('flow'):
            vph = float(flow.get("vehsPerHour"))
            flow.set("vehsPerHour", str(vph + random.gauss(0, FLOW_STD)))
        new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
        rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)
        
    else: # change parameters and add noise on flow
        rou_tree = ET.parse(f"{SCENARIO}.rou.xml")
        rou_root = rou_tree.getroot()
        for vtype in rou_root.findall('vType'):
            if vtype.get('id') == 'trial':
                for key, val in param.items():
                    vtype.set(key, str(val))
                break
        for flow in rou_root.findall('flow'):
            vph = float(flow.get("vehsPerHour"))
            flow.set("vehsPerHour", str(vph + random.gauss(0, FLOW_STD)))
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


class Discriminator(nn.Module):
    def __init__(self, traffic_shape, od_shape):
        super(Discriminator, self).__init__()

        self.num_samples, self.space, self.time, self.features = traffic_shape
        self.num_samples_od, self.routes, self.time_intervals = od_shape

        # ==================== Traffic Pattern Branch ====================
        self.traffic_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.features, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # Use actual output size after convolution and pooling for traffic features
        # self.traffic_output_size = 64 * (self.space // 2) * (self.time // 1)  # incorrect
        self.traffic_output_size = 1024  # Since the shape of traffic_features is [20, 960]

        # ==================== OD Matrix Branch ====================
        self.od_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.routes, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        self.od_output_size = 64 * (self.time_intervals // 2)

        # ==================== Fully Connected Layers ====================
        # Adjust input size based on actual combined size
        self.fc = nn.Sequential(
            nn.Linear(self.traffic_output_size + self.od_output_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, traffic_pattern, od_matrix):
        # ==================== Traffic Pattern Branch ====================
        traffic_pattern = traffic_pattern.permute(0, 3, 1, 2).unsqueeze(-1)  # Add dummy depth dimension
        traffic_features = self.traffic_conv(traffic_pattern)
        
        # ==================== OD Matrix Branch ====================
        od_matrix = od_matrix.permute(0, 1, 2)
        od_features = self.od_conv(od_matrix)

        # ==================== Combine Branches ====================
        combined = torch.cat([traffic_features, od_features], dim=1)

        # ==================== Final Output ====================
        output = self.fc(combined)
        return output


# ================ Discriminator Training Function ====================
def train_discriminator(discriminator, real_traffic_patterns, real_od_matrices, simulated_traffic_patterns, simulated_od_matrices):
    """
    Train the discriminator on real and simulated traffic patterns.
    traffic_patterns:(sample_size, time, space, features). 3 features (flow, density, speed), 
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


    # Generate training data
    # clear_directory("temp")
    # num_samples = 20
    # generate_real_data(num_samples=num_samples)
    # generate_sim_data(num_samples=num_samples)

    # Load data for discriminator training
    real_traffic_patterns = torch.load(f"../../data/{SCENARIO}/real/traffic_patterns.pt")
    real_od_matrices = torch.load(f"../../data/{SCENARIO}/real/demand.pt")
    sim_traffic_patterns = torch.load(f"../../data/{SCENARIO}/sim/traffic_patterns.pt")
    sim_od_matrices = torch.load(f"../../data/{SCENARIO}/sim/demand.pt")

    # Initialize the discriminator
    discriminator = Discriminator(real_traffic_patterns.shape, real_od_matrices.shape)
    output = discriminator(real_traffic_patterns, real_od_matrices)
    print(output.shape)
    print(output)

    # # Run optimization
    # sampler = optuna.samplers.TPESampler(seed=10)
    # study = optuna.create_study(direction='minimize', sampler=sampler)
    # study.optimize(objective, n_trials=N_TRIALS, n_jobs=os.cpu_count()-1)

    # # Save results
    # print('Best parameters:', study.best_params)

    