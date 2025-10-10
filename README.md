# FSDP using DeepSpeed for Vision Transformer Fine-Tuning

## Project Overview

This project focuses on implementing Fully Sharded Data Parallel (FSDP) using DeepSpeed to fine-tune a pre-trained Vision Transformer (ViT) on the CIFAR-100 dataset. The project demonstrates advanced distributed training techniques, memory optimization through model sharding, and efficient large-scale model training capabilities.


## Objectives

- Implement Fully Sharded Data Parallel (FSDP) using DeepSpeed for efficient model training
- Fine-tune a pre-trained Vision Transformer on CIFAR-100 dataset with memory optimization
- Demonstrate advanced distributed training techniques and model sharding
- Compare FSDP performance against standard training approaches
- Optimize training efficiency using mixed precision, ZeRO-Offload, and gradient checkpointing

## Dataset

- **Primary Dataset:** CIFAR-100 (Testing on other datasets as well)
- **Classes:** 100 fine-grained classes
- **Original Resolution:** 32x32 pixels
- **Target Resolution:** 224x224 pixels (ViT input requirement)

## Technical Requirements

### Core Technologies
- **Model:** Pre-trained Vision Transformer (google/vit-base-patch16-224)
- **Framework:** PyTorch, Transformers (Hugging Face)
- **Distributed Training:** DeepSpeed with FSDP (Fully Sharded Data Parallel)
- **Mixed Precision:** FP16/BF16
- **Memory Management:** ZeRO-Offload, Gradient Checkpointing, Model Sharding

### Key Features
- FSDP implementation for efficient model sharding across devices
- Mixed precision training for computational efficiency
- Advanced memory management through model parameter sharding
- Scalable distributed training capabilities
- Comprehensive logging and evaluation

## Project Tasks

### 1. Data Preparation
- [ ] Upsample CIFAR-100 images from 32x32 to 224x224 resolution
- [ ] Implement proper normalization for ViT input
- [ ] Apply data augmentation techniques
- [ ] Create train/validation/test splits

### 2. Model Setup
- [ ] Load pre-trained Vision Transformer from Hugging Face Hub
- [ ] Adapt model for CIFAR-100 classification (100 classes)
- [ ] Configure model parameters and architecture

### 3. FSDP Implementation with DeepSpeed
- [ ] Configure DeepSpeed with FSDP for model sharding
- [ ] Implement mixed precision training (FP16/BF16)
- [ ] Configure ZeRO-Offload for memory efficiency
- [ ] Enable gradient checkpointing
- [ ] Set up distributed training environment

### 4. Training Pipeline
- [ ] Set up training loop with proper loss functions
- [ ] Implement learning rate scheduling
- [ ] Configure batch size and training epochs
- [ ] Log key metrics (loss, accuracy) during training

### 5. Evaluation and Analysis
- [ ] Calculate final test accuracy
- [ ] Generate confusion matrix
- [ ] Analyze model performance across classes
- [ ] Document training metrics and results

## Bonus Features (Optional)

### Performance Comparison
- [ ] Compare FSDP training vs. standard training
- [ ] Benchmark runtime performance and memory efficiency
- [ ] Analyze peak GPU memory usage differences
- [ ] Measure model sharding effectiveness

### Multi-GPU Scaling with FSDP
- [ ] Implement FSDP across multiple GPUs
- [ ] Benchmark scaling efficiency with model sharding
- [ ] Measure speedup from 1 to 2+ GPUs
- [ ] Analyze FSDP distributed training performance
- [ ] Compare FSDP vs. traditional data parallel approaches

## Project Structure

```
distributed_vision_training/
├── README.md
├── src/
│   ├── data_preparation.py
│   ├── model_setup.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── models/
│   ├── logs/
│   └── plots/
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd distributed_vision_training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install DeepSpeed:
```bash
pip install deepspeed
```

## Usage

### Basic Training
```bash
python src/training.py --config configs/training_config.yaml
```

### Evaluation
```bash
python src/evaluation.py --model_path results/models/best_model.pt
```

## Expected Deliverables

1. **Code Implementation**
   - Complete training pipeline with DeepSpeed integration
   - Data preprocessing and augmentation scripts
   - Model evaluation and analysis tools

2. **Results and Analysis**
   - Final test accuracy on CIFAR-100
   - Confusion matrix visualization
   - Training metrics and loss curves
   - Performance comparison reports (if bonus features implemented)

3. **Documentation**
   - Comprehensive README with setup instructions
   - Code comments and docstrings
   - Analysis notebook with insights

## Key Metrics to Track

- **Accuracy:** Overall test accuracy on CIFAR-100
- **Memory Usage:** Peak GPU memory consumption with FSDP sharding
- **Training Time:** Total training duration and throughput
- **Convergence:** Loss curves and learning progress
- **Class Performance:** Per-class accuracy analysis
- **FSDP Efficiency:** Model sharding effectiveness and communication overhead
- **Scaling Performance:** Multi-GPU efficiency and speedup ratios

## Important things

- **FSDP Configuration:** Proper setup of model sharding and communication patterns
- **Memory Management:** Efficient handling of large transformer models with FSDP
- **Hyperparameter Tuning:** Learning rate, batch size, and optimization parameters
- **Distributed Training:** Managing communication overhead in FSDP implementation

## Future Enhancements

- Experiment with different ViT architectures
- Implement advanced augmentation techniques
- Explore knowledge distillation approaches
- Test on additional datasets beyond CIFAR-100

## Resources

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

**Note:** This project demonstrates advanced deep learning techniques and requires familiarity with PyTorch, transformers, and distributed training concepts.
