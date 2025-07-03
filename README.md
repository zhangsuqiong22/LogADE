# LogADE: Log Anomaly Detection and Explanation

A hybrid system combining graph-based deep learning and large language models for detecting and explaining anomalies in system logs.

## 1. Introduction

LogADE is a novel framework that leverages both graph neural networks (GNNs) and retrieval-augmented generation (RAG) to detect and explain anomalies in system logs. The system processes log data in two main stages:

1. **Graph-based Anomaly Detection**: Uses Directional Graph Convolutional Networks (DiGCN) to identify anomalous graph.

2. **LLM-powered Explanation Generation**: Uses a RAG system to retrieve similar anomaly cases and generate explanations for detected anomalies.

The framework is particularly effective for complex log analysis tasks, such as Kubernetes logs, where context and relationships between log entries are important.

## 2. Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Access to an LLM API (OpenAI GPT or Qwen)
- uv package manager (for virtual environment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/logade.git
cd logade
```

2. Set up a virtual environment using uv:
```bash
uv venv --python 3.10
source .venv/bin/activate 
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Configure API keys in `config.yaml`

## 3. Steps to Run

### Step 1: Prepare Data
1. Place your log data in CSV format in the `Data/` directory.
2. Generate graph representations from log data using:
```bash
python -m preprocessing.graph_generator --dataset YOUR_DATASET
```
   - `--dataset`: Name of your dataset (e.g., Kubelet, HDFS)
   - `--samples`: Number of graph samples to generate
   - `--anomaly_pct`: Percentage of anomalies in the samples (0.0-1.0)
   - `--adj_type`: Adjacency type (default: 'ib', options: 'un', 'appr', 'ib')

### Step 2: Run the Graph Level Anomaly Detection
```bash
python graphanomaly/main.py
```

This will:
- Process log data and build graph representations
- Train the DiGCN model to detect anomalies in log graphs
- Identify anomalous graphs and record the corresponding log lines


### Step 3: Run the RAG Pipeline
```bash
python main.py
```

This will:
- Generate explanations for these anomalous logs using the RAG system
- Output results to the `output/` directory
