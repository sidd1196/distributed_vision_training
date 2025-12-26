# Distributed Training for Vision Transformers and Large Language Models

A comprehensive project demonstrating distributed training strategies for both Vision Transformers (ViTs) and Large Language Models (LLMs) using DeepSpeed optimization, PyTorch DistributedDataParallel (DDP), and advanced model parallelism techniques.

## Project Overview

This project provides a complete framework for:

### Vision Transformer Training
- **Single-GPU Training**: LoRA-based fine-tuning with gradient checkpointing
- **Multi-GPU Training**: Distributed training using DDP, DeepSpeed ZeRO Stage 2, ZeRO Stage 3, and ZeRO++
- **Performance Analysis**: Profiling and trace analysis using PyTorch Profiler and HTA (Holistic Trace Analysis)

### Large Language Model Training
- **Progressive Model Parallelism**: From 1D (Data Parallelism) to 5D (Full Parallelism with MoE)
- **Comprehensive Metrics**: Model FLOPs Utilization (MFU), throughput, memory analysis
- **Industry-Standard Dataset**: MetaMathQA for math reasoning tasks
- **Profiling Integration**: Automatic trace collection for performance analysis

## Project Structure

```
.
├── advanceml-final-project-multi-gpu-testing-vit-huge.ipynb  # ViT-Huge multi-GPU experiments
├── LLM_Model_Parallelism_Qwen2.5_3B.ipynb                  # LLM parallelism (1D-5D)
├── metamathqa_utils.py                                     # MetaMathQA dataset utilities
├── DeepSpeed_trace_analysis.example.ipynb                  # HTA trace analysis
├── requirements.txt                                        # Python dependencies
└── README.md                                              # This file
```

## Key Features

### 1. **Vision Transformer Distributed Training**

- **DDP (DistributedDataParallel)**: Standard PyTorch multi-GPU training with full model replication
- **ZeRO Stage 2**: Partitions optimizer states and gradients across GPUs
- **ZeRO Stage 3**: Partitions optimizer states, gradients, and model parameters
- **ZeRO++**: Enhanced ZeRO Stage 3 with quantized gradient communication

### 2. **LLM Model Parallelism Strategies**

- **1D Parallelism**: Data Parallelism (DDP) - Baseline
- **2D Parallelism**: Data + Pipeline Parallelism
- **3D Parallelism**: Data + Pipeline + Tensor Parallelism (industry standard)
- **4D Parallelism**: Data + Pipeline + Tensor + Sequence Parallelism
- **5D Parallelism**: Full 5D with Expert Parallelism (MoE models)

### 3. **Memory Optimization Techniques**

- **Gradient Checkpointing**: Reduces memory during backward pass
- **Mixed Precision Training**: BF16/FP16 support for faster training and lower memory
- **CPU Offloading**: Optional offloading of optimizer states and parameters to CPU
- **Gradient Accumulation**: Supports large effective batch sizes with limited GPU memory

### 4. **Models and Datasets**

**Vision Transformers**:
- ViT-Base (~86M parameters) for single-GPU experiments
- ViT-Huge (~630M parameters) for multi-GPU experiments
- Datasets: CIFAR-100, Food-101

**Large Language Models**:
- Qwen 2.5 3B (~3B parameters) for parallelism experiments
- Dataset: MetaMathQA (395K math QA examples) - Industry-standard math reasoning dataset
- Used by MetaMath-Mistral-7B, OpenChat-3.5, CausalLM, and other industry models

### 5. **Performance Analysis**

- **PyTorch Profiler**: Detailed CPU/GPU activity traces with labeled operations
- **HTA (Holistic Trace Analysis)**: Advanced performance bottleneck identification
- **WandB Logging**: Real-time metrics, memory usage, throughput, and MFU tracking
- **Comprehensive Metrics**: Step time breakdown, communication overhead, scaling efficiency

## Prerequisites

1. **Hardware**: 
   - Multi-GPU system (tested with 2x Tesla T4, 1x A100)
   - CUDA-capable GPUs with sufficient memory
   - Recommended: 2-8 GPUs for LLM parallelism experiments

2. **Software**: 
   - Python 3.8+
   - PyTorch 2.0+ with CUDA support
   - DeepSpeed 0.10+
   - NCCL for multi-GPU communication
   - Transformers library for LLM support

3. **Dependencies**: Install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. **WandB Setup**: 
   - Create account at https://wandb.ai
   - Run `wandb login` or set `WANDB_API_KEY` environment variable
   - For Kaggle: Add `WANDB_API_KEY` to Secrets (Add-ons → Secrets)

## Quick Start

### Vision Transformer Training

See `advanceml-final-project-multi-gpu-testing-vit-huge.ipynb` for:
- DDP training with ViT-Huge on Food-101
- DeepSpeed ZeRO Stage 2, 3, and ZeRO++ experiments
- Profiling and trace collection

### LLM Model Parallelism

See `LLM_Model_Parallelism_Qwen2.5_3B.ipynb` for:
- Progressive parallelism strategies (1D → 5D)
- Qwen 2.5 3B model training on MetaMathQA
- Comprehensive metrics tracking (MFU, throughput, memory)
- Automatic profiling for trace analysis
- Scaling from 2 GPUs to 8 GPUs

**Running LLM Experiments**:
```bash
# 1D Parallelism (DDP)
torchrun --nproc_per_node=2 train_ddp_qwen3b_metamath.py

# 2D-5D Parallelism (DeepSpeed)
deepspeed --num_gpus=2 train_2d_pipeline_qwen3b_metamath.py
deepspeed --num_gpus=2 train_3d_parallelism_qwen3b_metamath.py
deepspeed --num_gpus=2 train_4d_parallelism_qwen3b_metamath.py
deepspeed --num_gpus=2 train_5d_parallelism_qwen3b_metamath.py
```

### Trace Analysis

See `DeepSpeed_trace_analysis.example.ipynb` for:
- Extracting and merging profiling traces
- HTA analysis for performance bottlenecks
- Visualizing GPU utilization and communication patterns
- Temporal breakdown and communication/computation overlap analysis

## LLM Model Parallelism Details

### Dataset: MetaMathQA

- **Size**: ~395K math QA examples
- **Format**: Instruction-following format compatible with Qwen 2.5 chat template
- **Utility Module**: `metamathqa_utils.py` provides consistent dataset loading across all experiments
- **Caching**: Tokenized dataset is cached for efficient distributed training

### Profiling Setup

Each LLM training script includes PyTorch Profiler that:
1. Profiles first 3 training steps (after 1 warmup step)
2. Saves Chrome traces to `./profiler_logs/llm_*_trace/` directories
3. Continues full training after profiling completes

**Trace files saved**:
- `./profiler_logs/llm_1d_ddp_trace/rank{rank}_trace.json`
- `./profiler_logs/llm_2d_pipeline_trace/rank{rank}_trace.json`
- `./profiler_logs/llm_3d_parallelism_trace/rank{rank}_trace.json`
- `./profiler_logs/llm_4d_parallelism_trace/rank{rank}_trace.json`
- `./profiler_logs/llm_5d_parallelism_trace/rank{rank}_trace.json`

### Metrics Tracked

All LLM experiments log comprehensive metrics to WandB:
- **Model FLOPs Utilization (MFU)**: GPU efficiency measurement
- **Throughput**: Samples/sec and tokens/sec
- **Memory Usage**: Allocated, reserved, and peak memory
- **Step Time Breakdown**: Forward, backward, optimizer, and total step time
- **Scaling Efficiency**: Comparison between 2 GPU and 8 GPU performance

## Experiment Tracking

All experiments automatically log to WandB with:
- Training/validation loss and accuracy
- GPU memory usage (allocated, reserved, peak)
- Training throughput (samples/second, tokens/second)
- Epoch/step duration
- System metrics (GPU utilization, temperature)
- Model FLOPs Utilization (MFU) for LLM experiments

## Performance Comparison

The project enables comparison of:

### Vision Transformers
- **Memory Efficiency**: Peak GPU memory usage across methods
- **Training Speed**: Throughput (samples/second) and epoch duration
- **Scalability**: Performance scaling with number of GPUs
- **Communication Overhead**: Communication time vs computation time

### Large Language Models
- **Parallelism Strategies**: 1D vs 2D vs 3D vs 4D vs 5D performance
- **Scaling Efficiency**: 2 GPU → 8 GPU scaling analysis
- **MFU Comparison**: GPU utilization across different parallelism approaches
- **Communication Patterns**: Analysis of AllGather, ReduceScatter, AllReduce operations
- **Pipeline Efficiency**: Bubble time analysis for pipeline parallelism

## Configuration Files

### DeepSpeed Configuration

DeepSpeed configurations are created programmatically within the notebooks. Configuration options include:
- ZeRO Stage 2, 3, and ZeRO++ optimization
- Pipeline parallelism (2D+ strategies)
- Tensor parallelism (3D+ strategies)
- BF16/FP16 mixed precision
- Gradient accumulation
- Optimizer and parameter offloading options

### LLM Parallelism Configuration

The LLM notebooks automatically configure parallelism based on GPU count:
- **2 GPUs**: 2 pipeline stages, 1-way tensor parallelism
- **8 GPUs**: 2 pipeline stages × 2-way tensor × 2 data parallel
- Custom configurations supported for other GPU counts

## Troubleshooting

### CUDA Out of Memory

1. Reduce `micro_batch_size` or `train_micro_batch_size_per_gpu`
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing
4. Use ZeRO Stage 3 with CPU offloading
5. For LLMs: Reduce sequence length or use smaller model variants

### NCCL Errors

1. Ensure NCCL is properly installed
2. Set `export NCCL_DEBUG=INFO` for detailed logs
3. Check network connectivity between GPUs
4. Verify CUDA version compatibility

### WandB Authentication

1. Run `wandb login` or set `WANDB_API_KEY`
2. In Kaggle: Add API key to Secrets
3. Check network connectivity to WandB servers

### LLM Training Issues

1. **Dataset Loading**: Ensure `metamathqa_utils.py` is in the same directory
2. **Model Download**: First run will download Qwen 2.5 3B model (~6GB)
3. **Memory**: 3B model requires significant GPU memory; adjust batch size accordingly
4. **Profiling**: Traces are saved automatically; check `./profiler_logs/` directory

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [HTA Documentation](https://github.com/facebookresearch/hta)
- [Qwen 2.5 Documentation](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [MetaMathQA Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (conceptual reference for 3D+ parallelism)


