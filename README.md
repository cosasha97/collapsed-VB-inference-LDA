# A Collapsed Variational Bayesian Inference Algorithm for Latent Dirichlet Allocation
Project for the course of Bayesian Machine learning - Master MVA

This work is based on the article **[1]** 'A Collapsed Variational Bayesian Inference Algorithm for Latent Dirichlet Allocation'. A detailed description of the work done can be found in the following [**report**](https://github.com/cosasha97/collapsed-VB-inference-LDA/blob/main/report.pdf). We coded the three methods proposed in **[1]** from scratch. The scripts of these methods can be found in the following files:
- vb.py:  variational Bayes method
- Gibbs.py collapsed Gibbs method
- collapsed_VB.py: collapsed variational Bayes method

## Generate the results
The notebook [**LDA_notebook.ipynb**](https://github.com/cosasha97/collapsed-VB-inference-LDA/blob/main/LDA_notebook.ipynb) can be followed step by step and shows the reader how to use the scripts of the different methods and generate the results presented in the report.

## Data
The 'Data/' folder contains KOS and Reuters datasets. 

## Results
Most results can quickly be generated from scratch. However, the collapsed VB algorithm is quite time consuming for large datasets. So to save the reader some time, we saved the outputs of this algorithm in the folder 'Results/'.

## Requirements
See requirements.txt

## References:
**[1] A Collapsed Variational Bayesian Inference Algorithm for Latent Dirichlet Allocation**
Teh Y.W., Newman D., and Welling, M.
Advances in Neural Information Processing Systems (NIPS) 19, 2007.
