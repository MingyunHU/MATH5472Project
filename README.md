# MATH5472Project
The code for MATH 5472 project "Summary and discussion of: 'Weighted Low Rank Matrix Approximation and Acceleration'." In the original paper (Tuzhilina et al. 2021), the proposed methods are implemented in R. I reproduce the experiments in MATLAB. The baseline and acceleration algorithms are tested on a small simulation data set and MovieLens 1M data.

## Requirements
- MATLAB R2019a

## How to run the code
Download `code` folder.

Run the codes in `simulation_hard.m` and `simulation_soft.m` to see the comparison of three algorithms for non-convex and convex WLRMA problems, respectively. The experiment is conducted on a simulation data set.

Run the code in `simulation_regular_Anderson.m` to see the comparison of Anderson and regularized-Anderson acceleration.

Run the code in `movielens.m` to view the comparison of three SVD-free algorithms for high dimension convex WLRMA problem. The experiment is conducted on MovieLens 1M data.
