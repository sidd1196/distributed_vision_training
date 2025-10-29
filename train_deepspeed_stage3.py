#!/usr/bin/env python3
"""
DeepSpeed ZeRO Stage 3 training script for ViT-Large on ImageNet-1k.
Uses BF16 mixed precision and W&B logging.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig, AutoImageProcessor
import wandb


def get_model(num_labels=1000):
    """Load ViT-Large model."""
    print("Loading google/vit-large-patch16-224 model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16
    )
    return model


def get_data_loaders(data_path, batch_size, world_size, rank, num_workers=4):
    """Create ImageNet data loaders."""
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load ImageNet from directory structure
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'val'),
        transform=train_transform
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def train_epoch(model_engine, train_loader, epoch, rank):
    """Train for one epoch with DeepSpeed."""
    model_engine.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(model_engine.device, non_blocking=True)
        labels = labels.to(model_engine.device, non_blocking=True)
        
        # Forward and backward pass (DeepSpeed handles autocast internally)
        loss = model_engine(images, labels=labels).loss
        
        # DeepSpeed backward pass
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model_engine, val_loader, rank):
    """Validate the model."""
    model_engine.eval()
    total_correct = 0
    total_samples = 0
    
    for images, labels in val_loader:
        images = images.to(model_engine.device, non_blocking=True)
        labels = labels.to(model_engine.device, non_blocking=True)
        
        outputs = model_engine(images)
        predictions = outputs.logits.argmax(dim=1)
        
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples * 100
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='DeepSpeed training for ViT-Large')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--deepspeed-config', type=str, default='deepspeed_config_stage3.json', help='DeepSpeed config file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    
    # Get rank and world size
    rank = deepspeed.comm.get_rank()
    world_size = deepspeed.comm.get_world_size()
    
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Starting DeepSpeed training on rank {rank}")
    
    # Initialize W&B (only on rank 0)
    if rank == 0:
        wandb.init(
            project='vit-imagenet-training',
            name='deepspeed-zero-stage3',
            config={
                'method': 'deepspeed-zero-stage3',
                'epochs': args.epochs,
                'deepspeed_config': args.deepspeed_config
            }
        )
    
    # Get model
    model = get_model()
    
    # Extract batch size from config
    import json
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    
    batch_size = ds_config.get('train_micro_batch_size_per_gpu', 8)
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        args.data_path,
        batch_size,
        world_size,
        rank,
        args.num_workers
    )
    
    if rank == 0:
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Micro batch size per GPU: {batch_size}")
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
        training_data=None  # We already have the loader
    )
    
    # Training loop
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
        
        # Set epoch for sampler
        train_loader.sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model_engine, train_loader, epoch, rank)
        
        # Validate
        val_accuracy = validate(model_engine, val_loader, rank)
        
        epoch_time = time.time() - epoch_start_time
        
        if rank == 0:
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Log to W&B
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_accuracy': val_accuracy,
                'epoch_time': epoch_time,
                'gpu_memory_allocated': torch.cuda.memory_allocated(rank) / 1e9,
                'gpu_memory_reserved': torch.cuda.memory_reserved(rank) / 1e9,
            })
    
    if rank == 0:
        print("\nTraining completed!")
        wandb.finish()


if __name__ == '__main__':
    main()

