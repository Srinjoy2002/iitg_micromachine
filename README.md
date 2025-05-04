
# Deep Learning–Driven Precision Micro-Machining and 3D Surface Profiling

An integrated hardware–software framework combining neural-network segmentation, fuzzy-logic motion control, and focus-stacking reconstruction to achieve sub-micron 3D surface metrology in micro-machining applications.

---

## Overview

This repository contains the design, implementation, and evaluation of an indigenous 3D profilometer. The system integrates:

- **Automated Z-axis traversal** with micron-level repeatability  
- **Neural-network segmentation** for in-focus region extraction  
- **Fuzzy-logic feedback control** over EtherCAT  
- **Variance-weighted focus stacking** for all-in-focus composite images  
- **3D point-cloud generation** and visualization  

Our motivation is to deliver a reliable, cost-effective solution for surface inspection in manufacturing, materials science, and related research domains.

---

## Key Features

1. **Hardware Platform**  
   - Precision linear actuation, real-time position feedback  
   - Synchronous image capture at calibrated focus intervals  
   - Modular mechanical and electrical interfaces  

2. **Machine Learning Module**  
   - ResNet–Attention U-Net architecture tailored to focus-stack microscopy  
   - Mean Intersection-over-Union (mIoU) of 0.92 on test datasets  
   - 45 ms average inference time per frame  

3. **Control System**  
   - Single Fuzzy Inference System (FocusFIS) for Z-axis error correction  
   - Closed-loop repeatability within ±0.1 µm  
   - Deterministic EtherCAT cycle times from 100 µs to 1 ms  

4. **3D Reconstruction**  
   - Focus variance improvement of 32% in composite images  
   - 3D surface RMSE < 0.8 µm against interferometric ground truth  
   - Exportable point-cloud and mesh formats for downstream analysis  
**Requirements**
1.Python 3.7+
2.OpenCV 4.5+
3.NumPy
4.Matplotlib
5.Open3D
6.glob
7.Custom scaling and measuring using mouse pointer

**Installation**
The following commands can download the SpiiPlus ACS Motion Control library:
```bash
python-- version
pip install SPiiPlusPython-{ppv}-cp{pv}-cp{abi}-{platform}.whl
```

 where the .whl files can be found in the same directory as a library. pv->Python Version

In the `motion_control_code` directory, you can find the following files:

- `final_gui_app.py` → Final GUI of Motion Control software  
- `enable.py`        → Supporting file for enabling the motor  
- `disable.py`       → Supporting file for disabling the motor  
- `Focus_app.py`     → Contains the focus-stacking and ML application  

Some other code for 3D visualization is located in the `supplementary_code` folder:

- `open3dviznew.py`  → Height-mapped colored 3D view  
- `matplotlib.py`    → Matplotlib-based 3D visualization  
- `real_view.py`     → Older version of `Focus_app.py`  
- `3DPCViz.py`       → Point-cloud-based visualization  
- `app.py`           → Test harness  

The images captured by the 3D profilometer are stored in:



The images that are being captured by the 3d profilometer are stored in “dataset/captured_img_date_time”.


