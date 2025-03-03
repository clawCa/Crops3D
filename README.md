# Crops3D
Utility Tools for the Crops3D Dataset


# Crops3D: A Diverse 3D Crop Dataset for Realistic Perception and Segmentation
![Crops3D](https://github.com/user-attachments/assets/8f7e5a85-c952-4964-b15e-85c3c163de47)

## Overview

Crops3D is a 3D crop dataset designed to support research in agricultural computer vision. It is derived from real-world agricultural scenarios and aims to facilitate advanced 3D point cloud analysis tasks. The dataset is characterized by its diversity, authenticity, and complexity, providing a resource for researchers and practitioners working in the agricultural domain.

## Key Features

- **Diversity**: Crops3D includes data from various point cloud acquisition methods and encompasses eight distinct crop types with a total of 1,230 samples.
- **Authenticity**: The dataset authentically represents crops in real-world agricultural settings, providing a realistic basis for research and development.
- **Complexity**: The intricate structures of the crops in Crops3D exhibit higher complexity compared to existing 3D public datasets, featuring substantial self-occlusion and increased complexity as crops mature.

## Supported Tasks

Crops3D is designed to support three critical tasks in 3D crop phenotyping:

1. **Instance Segmentation of Individual Plants**: Precise segmentation of individual plants in agricultural settings.
2. **Plant Type Perception**: Accurate identification and classification of different crop types.
3. **Plant Organ Segmentation**: Detailed segmentation of plant organs, enabling fine-grained analysis.

## Citation

If you use Crops3D in your research, please cite our paper:

```bibtex
@article{crops3d2024,
  title={Crops3D: A Diverse 3D Crop Dataset for Realistic Perception and Segmentation toward Agricultural Applications},
  author={Zhu, J. and Zhai, R. and Ren, H. et al.},
  journal={Scientific Data},
  year={2024},
  volume={11},
  number={1438},
  doi={10.1038/s41597-024-04290-0}
}
```

## Dataset Download

The dataset is stored in the [figshare](https://figshare.com/) database and can be accessed and downloaded using the following DOI: [https://doi.org/10.6084/m9.figshare.27313272](https://doi.org/10.6084/m9.figshare.27313272).

The dataset consists of a series of PLY and HDF5 files, compressed into a file named `Crops3D.zip`. It includes four main directories:

### 1. **Crops3D Directory**
   - Contains annotated raw point cloud data.
   - Includes eight directories named after specific crops: `Cabbage`, `Cotton`, `Maize`, `Potato`, `Rapeseed`, `Rice`, `Tomato`, and `Wheat`. Each directory stores PLY point cloud files for the respective crop.
   - Two text files list the relative paths of samples in the training and test sets.

### 2. **Crops3D_10k Directory**
   - Contains point cloud data subsampled to 10,000 points using FPS (Farthest Point Sampling).
   - The file structure mirrors that of the `Crops3D` directory for consistency and ease of use.

### 3. **Crops3D_10k-C Directory**
   - Contains a `corrupt` directory and two HDF5 (H5) files representing the training and test datasets.
   - The `corrupt` directory includes 36 H5 files: one clean test set and 35 files corresponding to seven types of corrupted test sets, each divided into five severity levels.

### 4. **Crops3D_IS Directory**
   - Contains point cloud data intended for individual plant segmentation at the plot scale (instance segmentation).
   - Includes one subdirectory and two text files. The subdirectory stores point cloud data for 50 plots, covering maize, potato, and rapeseed crops, formatted in PLY.
   - The two text files enumerate the relative paths of samples in the training and test sets.


