# Semantic Anomaly Stream Discovery

![Banner](src/assets/SASD_Banner.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Overview

An unsupervised vision pipeline that ingests video streams, identifies objects, and discovers anomalies through clustering in real-time.

## Visuals

![Demo](docs/images/demo.gif)

## Key Features

### Unsupervised Anomaly Discovery
Leverages DINOv2 features and HDBSCAN to automatically discover and segregate object categories from raw video feeds, eliminating the need for pre-defined labels or supervised training.

### Real-Time Flow Analytics
Combines high-performance SORT tracking with custom flow analysis to calculate velocities, active counts, and movement patterns for each discovered object cluster in real-time.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourrepo/semantic-object-stream-discovery.git
cd semantic-object-stream-discovery

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
