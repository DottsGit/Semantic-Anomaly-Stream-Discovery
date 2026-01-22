# Semantic Anomaly Stream Discovery

![Banner](src/assets/SASD_Banner.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Overview

An unsupervised vision pipeline that ingests video streams, identifies objects, and discovers anomalies through clustering in real-time.

https://github.com/user-attachments/assets/c8ccabc6-72b0-41af-a049-d8e15517f58c

https://github.com/user-attachments/assets/f5afa90e-02ee-426a-82c5-628642efb9e1

## Key Features

### Unsupervised Anomaly Discovery
Leverages YoloV8, DINOv2 features, and HDBSCAN to automatically discover objects and segregate anomalies from raw video feeds, eliminating the need for extensive pretraining.

### Real-Time Flow Analytics
Combines high-performance SORT tracking with custom flow analysis to calculate velocities, active counts, and movement patterns for each discovered object cluster in real-time.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourrepo/semantic-anomaly-stream-discovery.git
cd semantic-anomaly-stream-discovery

# Install dependencies
pip install -e .
```

### Usage

```bash
# Run on a video file
python -m src.main path/to/video.mp4 --warmup 60

# Run on a YouTube stream
python -m src.main "https://youtube.com/watch?v=..." --warmup 60
```
