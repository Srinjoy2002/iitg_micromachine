# Motor-Camera System Simulation with Fuzzy Logic Motor Regulation

This repository contains a MATLAB simulation of an integrated hardware system that combines a USB camera with a high-precision linear stage driven by a motor. The system uses fuzzy logic to regulate the motor’s 10-micron step movement based on encoder feedback, ensuring that the stage neither overshoots nor undershoots the commanded distance. In addition, the simulation animates the motor rotor rotation (at a nominal speed corresponding to 3000 rpm) and displays real-time feedback in a GUI.

> **Note:** This README focuses on the hardware and fuzzy logic aspects (motor regulation, encoder feedback, and rotor animation). The focus determination part (i.e., using image metrics) is simulated by a synthetic Gaussian focus function but is not described in detail here.

## Overview

The simulation models a hardware system with the following key components:
- **Linear Stage:** Simulated over 0 to 1000 microns in 10-micron steps (representing a 100 cm range with high resolution).
- **Encoder Feedback:** Simulated encoder updates the current position in microns.
- **Motor Rotor Animation:** A graphical representation (an arrow on a circle) rotates to mimic the motor’s rotor. The rotor rotates by 5° increments per capture step (scaled to represent a motor running at ~3000 rpm in real time).
- **Fuzzy Logic for Motor Regulation (MotorFIS):**  
  A fuzzy inference system (FIS) is designed to correct the motor step. The system simulates small random errors (±2 microns) in the motor’s step.  
  - **Input Variable:** `Error` (in microns), representing the difference between the actual movement (with error) and the commanded 10-micron step.  
    - **Range:** -2 to 2 microns  
    - **Membership Functions:**  
      - **Negative:** `trimf([-2 -2 0])` – Indicates undershoot (actual step < 10 microns).  
      - **Zero:** `trimf([-1 0 1])` – Indicates nearly perfect movement.  
      - **Positive:** `trimf([0 2 2])` – Indicates overshoot (actual step > 10 microns).  
  - **Output Variable:** `Adjustment` (in microns), representing the correction to be applied to the nominal step.  
    - **Range:** -2 to 2 microns  
    - **Membership Functions:**  
      - **Increase:** `trimf([-2 -2 0])` – If undershoot, the system increases the step size.  
      - **NoChange:** `trimf([-0.5 0 0.5])` – No adjustment if error is negligible.  
      - **Decrease:** `trimf([0 2 2])` – If overshoot, the system decreases the step size.  
  - **Fuzzy Rules:**  
    1. If **Error** is *Negative*, then **Adjustment** is *Increase*.  
    2. If **Error** is *Zero*, then **Adjustment** is *NoChange*.  
    3. If **Error** is *Positive*, then **Adjustment** is *Decrease*.  

- **GUI and Simulation Flow:**  
  - **Scanning Phase:** The stage scans from top to bottom (0–1000 microns) in 10-micron steps. At each step, a synthetic focus measure (Gaussian) is computed. The fuzzy system (FocusFIS) is used to decide if that position is “in focus.” The upper bound (UB) and lower bound (LB) of the in-focus region are determined.
  - **Capture Phase:** The stage then moves from UB to LB in nominal 10-micron steps. At each step, a random error is added to simulate motor inaccuracy. The MotorFIS evaluates the error and outputs an adjustment so that the effective step is corrected. Simultaneously, the encoder reading, stage marker, and a simulated camera image are updated. The motor rotor animation is updated by rotating an arrow by 5° per capture step.
  - At the end of the simulation, fuzzy membership function graphs for both FocusFIS and MotorFIS are saved as PNG files, and the fuzzy rules are printed to the MATLAB command window for documentation.

## MATLAB Blocks and Tools Used

- **GUI Components:**  
  - `uifigure`, `uiaxes`, `uilabel`, `uibutton` – for creating the user interface.
  
- **Timer Functions:**  
  - `timer` objects are used to simulate asynchronous scanning and capturing phases.

- **Image Display:**  
  - `imshow` – to display the simulated camera image (a grayscale image with brightness determined by the focus metric).

- **Fuzzy Logic Toolbox:**  
  - `mamfis` – to create a Mamdani fuzzy inference system.
  - `addInput`, `addOutput`, `addMF`, and `addRule` – to define fuzzy variables, membership functions, and rules.
  - `evalfis` – to evaluate the fuzzy inference system for a given input.

- **Plotting and Animation:**  
  - Standard plotting functions (`plot`, `quiver`, etc.) are used to animate the stage marker and motor rotor.

## Hardware Setup (Simulated)

- **Linear Stage:**  
  The stage is simulated over a 1000-micron range with 10-micron resolution. In a real hardware setup, a precision linear stage with a high-resolution encoder would be used.

- **Encoder:**  
  The encoder provides real-time position feedback in microns. In this simulation, the encoder value is updated on a label in the GUI.

- **Motor and Rotor:**  
  The motor is simulated by moving the stage. The motor rotor’s rotation is represented by a rotating arrow (updated by 5° increments per step) to emulate a motor running at approximately 3000 rpm.

- **Fuzzy Control Hardware Integration:**  
  In the actual system, a fuzzy logic controller would process the encoder data to ensure that each commanded 10-micron step is accurately executed. If the stage moves 9 microns, the controller would adjust the next command to add 1 micron; if it moves 11 microns, it would reduce the next step by 1 micron. This regulation is implemented in the simulation via MotorFIS.

## Installation and Running Instructions

1. **Requirements:**  
   - MATLAB 2024b  
   - Fuzzy Logic Toolbox  
   - Image Processing Toolbox (for `imshow`)  
   

