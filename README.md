# SHE-Nexus Reference Implementation

Version: 1.2.0  
Year first available: 2026  
Developer: Jiaoyang Wang  
Contact: wangjiaoyang1225@163.com  

## Requirements

- Programming language: Python 3.10 or newer
- Dependencies: NumPy >=1.24,<3; pandas >=2.0,<4
- Hardware: A standard 64-bit workstation running Windows, macOS or Linux
- Program size: Approximately 90 kB
- Example input: Approximately 26 kB, in JSON format

## Availability and license

The source code and synthetic example input are available in this repository at no charge. No reuse license is currently granted; public access does not itself grant permission to modify or redistribute the software.

## Run

Keep the supplied JSON input file in the same directory as `SHE_Nexus.py`, then run:

```bash
python -m pip install "numpy>=1.24,<3" "pandas>=2.0,<4"
python SHE_Nexus.py
```

The program outputs `period_performance_and_feedback.csv`, which summarizes period-level performance and feedback.

## Scope and data

This implementation provides the coupled reservoir-operation and feedback workflow. It does not include the VIC model or case-study calibration procedures. 

All example inputs are synthetic and are provided solely to demonstrate the computational workflow. They do not reproduce the case-study findings. Confidential observations and case-specific reservoir parameters are not included.

## Citation

If you use this software or its methods in your research, please cite the accompanying manuscript:

Dynamic feedback-driven modelling and synergistic optimization of the water supply–hydropower generation–environment conservation nexus with inter-basin water diversion projects.

Manuscript under review. Full publication details and the DOI will be added when available.
