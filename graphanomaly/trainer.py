#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer classes for graph-based anomaly detection models
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base trainer class that defines the common interface
    """
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
    def train(self, train_loader):
        """Train for one epoch"""
        raise NotImplementedError
        
    def test(self, test_loader):
        """Evaluate model on test set"""
        raise NotImplementedError
        
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


class MeanTrainer(BaseTrainer):
    """
    Trainer for one-class classification using Deep SVDD approach
    with mean aggregation for graph embeddings
    
    Parameters:
    -----------
    model : torch.nn.Module
        Graph neural network model
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    alpha : float
        Weight for SVDD loss
    beta : float
        Weight for regularization loss
    device : torch.device
        Device to run training on
    regularizer : str
        Type of regularization to use ('variance' or 'l2')
    """
    def __init__(self, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
        super(MeanTrainer, self).__init__(model, optimizer, device)
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer
        self.center = None  # SVDD center will be computed during first epoch
        
    def train(self, train_loader):
        """
        Train the model for one epoch
        
        Parameters:
        -----------
        train_loader : torch_geometric.data.DataLoader
            DataLoader containing training graphs
            
        Returns:
        --------
        float
            Average SVDD loss for the epoch
        """
        self.model.train()
        total_loss = 0
        svdd_losses = []
        reg_losses = []
        
        # Process each batch
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass through the model
            embeddings = self.model(batch)
            
            # Compute mean embeddings for each graph
            mean_embeddings = []
            for emb in embeddings:
                if emb.size(0) > 0:  # Ensure there are nodes in the graph
                    mean_emb = torch.mean(emb, dim=0)
                    mean_embeddings.append(mean_emb)
            
            if len(mean_embeddings) == 0:
                continue  # Skip empty batches
                
            mean_embeddings = torch.stack(mean_embeddings)
            
            # First epoch: compute center as mean of all embeddings
            if self.center is None:
                with torch.no_grad():
                    self.center = torch.mean(mean_embeddings, dim=0)
            
            # Compute SVDD loss (distance to center)
            svdd_loss = torch.mean(torch.sum((mean_embeddings - self.center) ** 2, dim=1))
            
            # Compute regularization loss if beta > 0
            if self.beta > 0:
                if self.regularizer == "variance":
                    # Variance regularization
                    variance = torch.mean(torch.var(mean_embeddings, dim=0))
                    reg_loss = -variance  # Maximize variance (minimize negative variance)
                else:
                    # L2 regularization
                    reg_loss = torch.mean(torch.sum(mean_embeddings ** 2, dim=1))
                
                # Total loss
                loss = self.alpha * svdd_loss + self.beta * reg_loss
                reg_losses.append(reg_loss.item())
            else:
                # No regularization
                loss = svdd_loss
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            total_loss += loss.item()
            svdd_losses.append(svdd_loss.item())
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_svdd_loss = np.mean(svdd_losses)
        
        if self.beta > 0:
            avg_reg_loss = np.mean(reg_losses)
            logger.debug(f"Avg SVDD Loss: {avg_svdd_loss:.6f}, Avg Reg Loss: {avg_reg_loss:.6f}")
        
        return avg_svdd_loss
        
    def test(self, test_loader):
        """
        Evaluate model on test set
        
        Parameters:
        -----------
        test_loader : torch_geometric.data.DataLoader
            DataLoader containing test graphs
            
        Returns:
        --------
        float
            Average Precision score
        float
            ROC-AUC score
        torch.Tensor
            Distances for each test sample
        torch.Tensor
            Ground truth labels
        """
        self.model.eval()
        dists_list = []
        
        with torch.no_grad():
            # Process each batch
            for batch in test_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass through the model
                embeddings = self.model(batch)
                
                # Compute mean embeddings for each graph
                mean_embeddings = []
                for emb in embeddings:
                    if emb.size(0) > 0:  # Ensure there are nodes in the graph
                        mean_emb = torch.mean(emb, dim=0)
                        mean_embeddings.append(mean_emb)
                
                if len(mean_embeddings) == 0:
                    continue  # Skip empty batches
                    
                mean_embeddings = torch.stack(mean_embeddings)
                
                # Compute distance to center
                dists = torch.sum((mean_embeddings - self.center) ** 2, dim=1)
                dists_list.append(dists)
            
            # Concatenate all distances and labels
            if len(dists_list) == 0:
                return 0.0, 0.0, torch.tensor([]), torch.tensor([])
                
            dists = torch.cat(dists_list)
            labels = torch.cat([batch.y for batch in test_loader])
            
            # Convert to numpy for scikit-learn metrics
            dists_np = dists.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Calculate evaluation metrics
            ap = average_precision_score(
                y_true=labels_np,
                y_score=dists_np,
                average=None,
                pos_label=1
            )
            
            roc_auc = roc_auc_score(
                y_true=labels_np,
                y_score=dists_np,
                average=None
            )
            
            return ap, roc_auc, dists, labels

    def get_radius(self, train_loader=None, percentile=95.0, sigma_multiplier=2.0):

            if self.center is None:
                raise ValueError("SVDD center has not been computed yet. Run training first.")
            
            if train_loader is None:
                return 0.1  
            
            self.model.eval()
            distances = []
            
            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(self.device)
                    
                    embeddings = self.model(batch)
                    
                    mean_embeddings = []
                    for emb in embeddings:
                        if emb.size(0) > 0:  
                            mean_emb = torch.mean(emb, dim=0)
                            mean_embeddings.append(mean_emb)
                    
                    if len(mean_embeddings) == 0:
                        continue 
                        
                    mean_embeddings = torch.stack(mean_embeddings)
                    dists = torch.sum((mean_embeddings - self.center) ** 2, dim=1)
                    distances.append(dists)
            if len(distances) == 0:
                logger.warning("No distances collected for radius calculation. Using default.")
                return 0.1
                
            distances = torch.cat(distances).cpu().numpy()
            
            radius_percentile = np.percentile(distances, percentile)

            radius_sigma = distances.mean() + sigma_multiplier * distances.std()
            
            radius_mean = distances.mean()
            radius = radius_mean
            
            logger.info(f"Calculated radius: {radius:.6f} (mean: {radius_mean:.6f}, "
                    f"percentile-{percentile}: {radius_percentile:.6f}, "
                    f"mean+{sigma_multiplier}*std: {radius_sigma:.6f})")
            
            return radius
    
class MaxTrainer(MeanTrainer):
    """
    Trainer for one-class classification using Deep SVDD approach
    with max aggregation for graph embeddings
    """
    def train(self, train_loader):
        """Training with max pooling instead of mean pooling"""
        self.model.train()
        total_loss = 0
        svdd_losses = []
        reg_losses = []
        
        # Process each batch
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass through the model
            embeddings = self.model(batch)
            
            # Compute max embeddings for each graph
            max_embeddings = []
            for emb in embeddings:
                if emb.size(0) > 0:  # Ensure there are nodes in the graph
                    max_emb = torch.max(emb, dim=0)[0]  # Get values, not indices
                    max_embeddings.append(max_emb)
            
            if len(max_embeddings) == 0:
                continue  # Skip empty batches
                
            max_embeddings = torch.stack(max_embeddings)
            
            # First epoch: compute center as mean of all embeddings
            if self.center is None:
                with torch.no_grad():
                    self.center = torch.mean(max_embeddings, dim=0)
            
            # Compute SVDD loss (distance to center)
            svdd_loss = torch.mean(torch.sum((max_embeddings - self.center) ** 2, dim=1))
            
            # Compute regularization loss if beta > 0
            if self.beta > 0:
                if self.regularizer == "variance":
                    # Variance regularization
                    variance = torch.mean(torch.var(max_embeddings, dim=0))
                    reg_loss = -variance  # Maximize variance (minimize negative variance)
                else:
                    # L2 regularization
                    reg_loss = torch.mean(torch.sum(max_embeddings ** 2, dim=1))
                
                # Total loss
                loss = self.alpha * svdd_loss + self.beta * reg_loss
                reg_losses.append(reg_loss.item())
            else:
                # No regularization
                loss = svdd_loss
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            total_loss += loss.item()
            svdd_losses.append(svdd_loss.item())
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_svdd_loss = np.mean(svdd_losses)
        
        return avg_svdd_loss
    
    def test(self, test_loader):
        """Testing with max pooling instead of mean pooling"""
        self.model.eval()
        dists_list = []
        
        with torch.no_grad():
            # Process each batch
            for batch in test_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass through the model
                embeddings = self.model(batch)
                
                # Compute max embeddings for each graph
                max_embeddings = []
                for emb in embeddings:
                    if emb.size(0) > 0:  # Ensure there are nodes in the graph
                        max_emb = torch.max(emb, dim=0)[0]  # Get values, not indices
                        max_embeddings.append(max_emb)
                
                if len(max_embeddings) == 0:
                    continue  # Skip empty batches
                    
                max_embeddings = torch.stack(max_embeddings)
                
                # Compute distance to center
                dists = torch.sum((max_embeddings - self.center) ** 2, dim=1)
                dists_list.append(dists)
            
            # Rest of the method is same as in MeanTrainer
            if len(dists_list) == 0:
                return 0.0, 0.0, torch.tensor([]), torch.tensor([])
                
            dists = torch.cat(dists_list)
            labels = torch.cat([batch.y for batch in test_loader])
            
            # Convert to numpy for scikit-learn metrics
            dists_np = dists.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Calculate evaluation metrics
            ap = average_precision_score(
                y_true=labels_np,
                y_score=dists_np,
                average=None,
                pos_label=1
            )
            
            roc_auc = roc_auc_score(
                y_true=labels_np,
                y_score=dists_np,
                average=None
            )
            
            return ap, roc_auc, dists, labels


class SumTrainer(MeanTrainer):
    """
    Trainer for one-class classification using Deep SVDD approach
    with sum aggregation for graph embeddings
    """
    def train(self, train_loader):
        """Training with sum pooling instead of mean pooling"""
        self.model.train()
        total_loss = 0
        svdd_losses = []
        reg_losses = []
        
        # Process each batch
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass through the model
            embeddings = self.model(batch)
            
            # Compute sum embeddings for each graph
            sum_embeddings = []
            for emb in embeddings:
                if emb.size(0) > 0:  # Ensure there are nodes in the graph
                    sum_emb = torch.sum(emb, dim=0)
                    sum_embeddings.append(sum_emb)
            
            if len(sum_embeddings) == 0:
                continue  # Skip empty batches
                
            sum_embeddings = torch.stack(sum_embeddings)
            
            # First epoch: compute center as mean of all embeddings
            if self.center is None:
                with torch.no_grad():
                    self.center = torch.mean(sum_embeddings, dim=0)
            
            # Compute SVDD loss (distance to center)
            svdd_loss = torch.mean(torch.sum((sum_embeddings - self.center) ** 2, dim=1))
            
            # Compute regularization loss if beta > 0
            if self.beta > 0:
                if self.regularizer == "variance":
                    # Variance regularization
                    variance = torch.mean(torch.var(sum_embeddings, dim=0))
                    reg_loss = -variance  # Maximize variance (minimize negative variance)
                else:
                    # L2 regularization
                    reg_loss = torch.mean(torch.sum(sum_embeddings ** 2, dim=1))
                
                # Total loss
                loss = self.alpha * svdd_loss + self.beta * reg_loss
                reg_losses.append(reg_loss.item())
            else:
                # No regularization
                loss = svdd_loss
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            total_loss += loss.item()
            svdd_losses.append(svdd_loss.item())
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_svdd_loss = np.mean(svdd_losses)
        
        return avg_svdd_loss
    
    def test(self, test_loader):
        """Testing with sum pooling instead of mean pooling"""
        self.model.eval()
        dists_list = []
        
        with torch.no_grad():
            # Process each batch
            for batch in test_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass through the model
                embeddings = self.model(batch)
                
                # Compute sum embeddings for each graph
                sum_embeddings = []
                for emb in embeddings:
                    if emb.size(0) > 0:  # Ensure there are nodes in the graph
                        sum_emb = torch.sum(emb, dim=0)
                        sum_embeddings.append(sum_emb)
                
                if len(sum_embeddings) == 0:
                    continue  # Skip empty batches
                    
                sum_embeddings = torch.stack(sum_embeddings)
                
                # Compute distance to center
                dists = torch.sum((sum_embeddings - self.center) ** 2, dim=1)
                dists_list.append(dists)
            
            # Rest of the method is same as in MeanTrainer
            if len(dists_list) == 0:
                return 0.0, 0.0, torch.tensor([]), torch.tensor([])
                
            dists = torch.cat(dists_list)
            labels = torch.cat([batch.y for batch in test_loader])
            
            # Convert to numpy for scikit-learn metrics
            dists_np = dists.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Calculate evaluation metrics
            ap = average_precision_score(
                y_true=labels_np,
                y_score=dists_np,
                average=None,
                pos_label=1
            )
            
            roc_auc = roc_auc_score(
                y_true=labels_np,
                y_score=dists_np,
                average=None
            )
            
            return ap, roc_auc, dists, labels


def get_trainer(aggregation, model, optimizer, alpha, beta, device):
    """
    Factory function to get the appropriate trainer based on aggregation type
    
    Parameters:
    -----------
    aggregation : str
        Type of aggregation ('Mean', 'Max', or 'Sum')
    model : torch.nn.Module
        GNN model
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    alpha : float
        Weight for SVDD loss
    beta : float
        Weight for regularization loss
    device : torch.device
        Device to run training on
        
    Returns:
    --------
    BaseTrainer
        Trainer instance
    """
    if aggregation == "Mean":
        return MeanTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
        )
    elif aggregation == "Max":
        return MaxTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
        )
    elif aggregation == "Sum":
        return SumTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
        )
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation}")