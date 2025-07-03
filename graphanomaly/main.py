#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Logs2Graph
"""
import argparse
import os
import yaml
import torch
import numpy as np
from datetime import datetime
import pickle
import logging
from preprocessing.graph_generator import generate_graphs
from digcn import DiGCN, DiGCN_IB_Sum

from trainer import get_trainer
from dataloader import create_loaders

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Logs2Graph: Log Anomaly Detection with GNNs')
    
    # Dataset configuration
    parser.add_argument('--data', type=str, default='Kubelet',
                        choices=['HDFS', 'OpenStack', 'Kubelet', 'BGL', 'Thunderbird'],
                        help='Dataset to use')
    parser.add_argument('--regenerate_graphs', action='store_true',
                        help='Whether to regenerate graphs from log data')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='DiGCN',
                        choices=['DiGCN', 'GIN', 'DiGCN_IB_Sum'],
                        help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to use bias terms in the GNN')
    parser.add_argument('--aggregation', type=str, default='Mean',
                        choices=['Mean', 'Max', 'Sum'],
                        help='Type of graph level aggregation')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay regularization')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha parameter for training')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='Beta parameter for training')
    
    # System configuration
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID, -1 for CPU')
    parser.add_argument('--data_seed', type=int, default=421,
                        help='Random seed for data splitting')
    parser.add_argument('--model_seed', type=int, default=0,
                        help='Random seed for model initialization')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    return parser.parse_args()

def save_results(output_path, results, args, best_epoch):
    """
    Save experiment results and configuration to disk
    
    Parameters:
    -----------
    output_path : str
        Path where to save the results
    results : list
        List of dictionaries containing results for each epoch
    args : dict
        Dictionary of arguments/parameters used for the experiment
    best_epoch : int
        Index of the best epoch
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data to save
    data = {
        'results': results,
        'args': args,
        'best_epoch': best_epoch,
        'best_result': results[best_epoch],
    }
    
    # Convert PyTorch tensors to CPU numpy arrays for better serialization
    for epoch_result in data['results']:
        for key, value in epoch_result.items():
            if isinstance(value, torch.Tensor):
                epoch_result[key] = value.cpu().numpy()
    
    # Save to disk
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try alternate location if the original path fails
        alternate_path = os.path.join(os.path.expanduser("~"), "logs2graph_results.pkl")
        try:
            with open(alternate_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Results saved to alternate location: {alternate_path}")
        except Exception as e2:
            print(f"Failed to save results to alternate location: {e2}")
    
    # Also save a summary text file for quick reference
    try:
        summary_path = output_path.replace('.pkl', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Experiment Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Model: {args['model']}\n")
            f.write(f"Dataset: {args['data']}\n")
            f.write(f"Aggregation: {args['aggregation']}\n\n")
            
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"SVDD loss: {results[best_epoch]['svdd_loss']:.6f}\n")
            f.write(f"AP: {results[best_epoch]['ap']:.6f}\n")
            f.write(f"ROC-AUC: {results[best_epoch]['roc_auc']:.6f}\n\n")

            f.write(f"Top Detected Anomalies:\n")
            f.write(f"=====================\n\n")
            
            if 'anomaly_ids' in results[best_epoch]:
                anomalies = results[best_epoch]['anomaly_ids']
                for i, anomaly in enumerate(anomalies):
                    f.write(f"#{i+1}: ID: {anomaly['id']}\n")
                    f.write(f"  - Distance: {anomaly['distance']:.6f}\n")
                    f.write(f"  - True Label: {'Anomaly' if anomaly['true_label'] == 1 else 'Normal'}\n")
                    f.write(f"  - Index: {anomaly['index']}\n\n")
            else:
                f.write("No anomaly detection information available.\n\n")            
            f.write(f"Configuration:\n")
            for key, value in args.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Summary saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary: {e}")

def setup_directories(dataset_name):
    """
    Set up necessary directories for the specified dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to set up directories for
    """
    # Define base directory - use environment variable if available, otherwise use current directory
    base_dir = os.environ.get('LOGS2GRAPH_ROOT', os.path.abspath('.'))
    
    # Create standard directories
    dirs = [
        os.path.join(base_dir, 'Data', dataset_name),
        os.path.join(base_dir, 'Data', dataset_name, 'Raw'),
        os.path.join(base_dir, 'Data', dataset_name, 'Graph'),
        os.path.join(base_dir, 'outputs'),
        os.path.join(base_dir, 'logs'),
    ]
    
    # Create each directory if it doesn't exist
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    return base_dir

def get_model(model_name, num_features, hidden_dim, num_layers, bias=False):
    if model_name == 'DiGCN':
        return DiGCN(
            nfeat=num_features,
            nhid=hidden_dim,
            nlayer=num_layers,
            bias=bias
        )
    elif model_name == 'DiGCN_IB_Sum':
        from DataLoader import DiGCN_IB_Sum
        return DiGCN_IB_Sum(
            nfeat=num_features,
            nhid=hidden_dim,
            nlayer=num_layers,
            bias=bias
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_radius(self):

    self.model.eval()
    
    embeddings = []
    with torch.no_grad():
        for data in self.train_data:  
            data = data.to(self.device)
            embedding = self.model(data)
            embeddings.append(embedding)
    
    embeddings = torch.cat(embeddings, dim=0)
    distances = torch.sum((embeddings - self.center) ** 2, dim=1)
    
    radius = distances.mean().item()
    
    # 方法2: 使用平均距离 + 标准差的倍数
    # std_dev = distances.std().item()
    # radius = distances.mean().item() + 2 * std_dev
    
    return radius
   
def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    # Set random seeds
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    
    # Generate graphs if needed
    if args.regenerate_graphs:
        print(f"Generating graphs for {args.data} dataset...")
        generate_graphs(
            dataset=args.data,
            num_samples=10000,  # Can be parameterized
            anomaly_percentage=0.05  # Dataset-specific, can be parameterized
        )
    
    # Create data loaders
    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(
        data_name=args.data,
        batch_size=args.batch_size,
        dense=False,
        data_seed=args.data_seed
    )
    
    # Print dataset statistics
    print("Dataset statistics:")
    train_labels = np.array([data.y.item() for data in train_dataset])
    test_labels = np.array([data.y.item() for data in test_dataset])
    print(f"TRAIN: {len(train_dataset)} graphs, Normal: {(train_labels==0).sum()}, Anomaly: {(train_labels==1).sum()}")
    print(f"TEST: {len(test_dataset)} graphs, Normal: {(test_labels==0).sum()}, Anomaly: {(test_labels==1).sum()}")
    
    # Initialize model
    model = get_model(
        model_name=args.model,
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        bias=args.bias
    )
    
    # Initialize optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize trainer
    trainer = get_trainer(
        aggregation=args.aggregation,
        model=model,
        optimizer=optimizer,
        alpha=args.alpha,
        beta=args.beta,
        device=device
    )
    
    # Training loop
    results = []
    for epoch in range(args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        print("Training...")
        svdd_loss = trainer.train(train_loader=train_loader)
        print(f"SVDD loss: {svdd_loss:.6f}")
        
        # Evaluate
        print("Evaluating...")
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        print(f"ROC-AUC: {roc_auc:.6f}")
        
        # Store results
        results.append({
            'epoch': epoch,
            'svdd_loss': svdd_loss,
            'ap': ap,
            'roc_auc': roc_auc,
            'dists': dists,
            'labels': labels
        })
    
    # Find best model based on SVDD loss
    best_idx = np.argmin([r['svdd_loss'] for r in results[1:]])
    best_epoch = best_idx + 1
    
    print(f"Best model at epoch {best_epoch}:")
    print(f"  SVDD loss: {results[best_epoch]['svdd_loss']:.6f}")
    print(f"  AP: {results[best_epoch]['ap']:.6f}")
    print(f"  ROC-AUC: {results[best_epoch]['roc_auc']:.6f}")

    anomaly_ids = []


    radius = trainer.get_radius()
    #print(f"Detection radius: {radius:.6f}")


    for i, data in enumerate(test_dataset):
        if dists[i] > radius:  
            anomaly_ids.append({
                'index': i,
                'distance': dists[i].item(),
                'true_label': labels[i].item(),
                'id': str(i),  # 仅使用索引作为ID
                'radius': radius  # 记录用于检测的半径值
            })
    results[best_epoch]['anomaly_ids'] = anomaly_ids
        
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"{args.model}_{args.aggregation}_{args.data}_{timestamp}.pkl"
    )
    save_results(
        output_path=output_path,
        results=results,
        args=vars(args),
        best_epoch=best_epoch
    )
    
    return results, best_epoch

if __name__ == "__main__":
    main()
