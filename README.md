# Task 2 - Attitude and Navigation Reference System
Flight Dynamics Task 2 - Attitude and Navigation Reference System using Tello IMU data

Flight Dynamics course project.

## Authors
- Daniela Miranda
- Alejandro Pimienta
- David Alejandro Díaz

# Objective
This project implements a basic inertial navigation and attitude visualization tool using raw IMU measurements from a Tello drone.

The program:
- propagates aircraft attitude using angular-rate measurements,
- computes velocity and position in the NED frame,
- visualizes trajectory, Euler angles, quaternions, and angle-axis representation.

# Files
- `waze.py`: main Python script
- `tello_imu_example.csv`: raw IMU input data
- `tello_ground_truth.csv`: reference trajectory for comparison
- `trayectoria.gif`: animated XY trajectory
- `requirements.txt`: required Python libraries

# Required Libraries
numpy
pandas
matplotlib
pillow

# Installation
Install the required libraries with:

```bash
pip install -r requirements.txt
