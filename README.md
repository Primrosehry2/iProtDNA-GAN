# iProtDNA-GAN
Pytorch implementation of paper 'iProtDNA-GAN: Prediction of Protein-DNA Binding Sites Based on Imbalanced Graph Neural Networks' 

## Dependencies
- python 3.8
- pytorch 1.11.0
- torchvision 0.12.0
- numpy

## Repo for iProtDNA-GAN framework
This repo holds the code of iProtDNA-GAN framework for protein-ligands binding sites prediction. Five processed datasets are published, including TR646, TR573, TE46, TE129, and TE181.

iProtDNA-GAN is primarily dependent on a large-scale pre-trained protein language model ESM2 implemented using PyTorch. Please install the dependencies in advance.

## Files and folders description
### 1. Raw_data
This folder contains raw data. The first line is the protein id (which might not be PDB ID); the second line is the protein sequence, and the third line is the data label indicating binding sites or non-binding sites.

Note: if you wanna find the original data in PDB format, please kindly refer to the following 3 papers: DBPred, GraphBind, and GraphSite.

### 2. Weights
This folder contains trained weight files. Specifically, 646_46.pt corresponds to the trained model for task TR646, suitable for testing the TE46 dataset. 573_129_181.pt represents the trained model for task TR573, applicable for testing the TE129 dataset. Lastly, 573_129_181.pt denotes another trained model for task TR573, intended for testing the TE181 dataset.

## Codes description
The training and implementation of this project are based on PyTorch 1.11.0 and PyTorch-lightning 2.0.3, the higher version might not be compatible. 

### 1. data_processing.py
Using Generative Adversarial Networks to Generate Balanced Data

### 2. model.py
Implementation of backbone models such as Generator and Discriminator Used for Generating Balanced Graphs with Generative Adversarial Networks, Graphsage, MLP.

### 3. utils.py
Processing of balanced datasets, and Focal Loss 

### 4. main.py
Implementation of model training. Here we provide a sample file, and please use iProtDNA-GAN as following commands:

### 5. test.py
Implementation of independent testing of the model. Here we provide a sample file, and please use iProtDNA-GAN as following commands:

```
python test.py --input 646_46.pt, 46.pt --output result
```

## Citation
If any problems occur via running this code, please contact us at 2220641288@qq.com.

Thank you!
