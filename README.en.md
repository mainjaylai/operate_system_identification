# Operating System Fingerprinting System

This is an operating system fingerprinting system based on network traffic analysis, which identifies different operating system types by analyzing the characteristics of network packets.

## Features

- Automatically generate various types of network traffic data
- Extract network traffic features from PCAP files
- Support identification of multiple operating systems (CentOS, Ubuntu, Debian)
- Use machine learning models for operating system classification
- Integrate multiple classifiers to improve recognition accuracy

## Project Structure

```
project/
├── send_https_requests.sh # Network traffic generation script
├── pcap_extractor.py # PCAP file feature extraction
├── process_features.py # Feature processing and data preprocessing
├── model_training.py # Model training and evaluation
├── feature_files/ # Directory for storing feature files
└── pcap_files/ # Directory for storing PCAP files
```

## Technology Stack

- Python 3.x
- scikit-learn
- pandas
- numpy
- pyshark
- loguru
- bash script

## Installation Instructions

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn pyshark loguru
```

2. Install system dependencies:

```bash
sudo apt-get install tcpdump tshark
```

## Usage

1. Generate network traffic data:

```bash
chmod +x send_https_requests.sh
./send_https_requests.sh
```

2. Extract features from PCAP files:

```bash
python pcap_extractor.py
```

3. Process feature data:

```bash
python process_features.py
```

4. Train the model:

```bash
python model_training.py
```

## Feature Description

The main features extracted by the system include:
- IP packet length statistical features (max, min, mean, std)
- TTL value statistical features
- TCP window size statistical features
- TCP flag statistics
- TCP timestamp features

## Model Description

Uses ensemble learning methods, including the following classifiers:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression

## Notes

1. Running the traffic generation script requires root privileges
2. Ensure sufficient disk space to store PCAP files
3. It is recommended to run the traffic generation script in a test environment

## License

MIT License

## Contribution Guide

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## Contact

For any issues, please submit an Issue or Pull Request.
