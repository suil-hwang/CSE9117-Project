# CSE9117 Course Project - Curvature-Adaptive Sampling for CWF

**Course**: CSE9117 Topics in Computer Graphics (2025 Spring, Hanyang University)

## Overview

This project improves the [CWF mesh simplification algorithm](https://arxiv.org/abs/2404.15661) by implementing **curvature-based adaptive sampling** to sample more densely in high-curvature regions, better preserving weak features.

Additionally, we worked on [Island-Preserving UV Transfer for Simplified Meshes](https://github.com/suil-hwang/island-preserving-uv-transfer) (maintained in a separate repository).


## Key Idea

- **Problem**: CWF's poisson-disk sampling misses important geometric details
- **Solution**: Sample more points in high-curvature regions

## Usage

```bash
# Sample points
python sampling/curvature_based_sampling.py mesh.obj -n 1000
```