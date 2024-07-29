<div align="center">
  
# CaLMPhosKAN

</div>


<p align="center">
  Predict General Phosphorylation Modification in Proteins Using the Fusion of Codon-Aware Embeddings and Amino Acid-Aware Embeddings with Wavelet-based Kolmogorov–Arnold Network
</p>

<p align="center">
<img src="images/example_run.png"/> 
</p>

## About

### Motivation

#### Codon-level Embeddings
The recent introduction of CaLM, a protein language model (pLM) trained on protein-coding DNA sequences, allows for extraction and use of information out of the codon space of proteins. This space has provided usefull information that allows CaLM to outperform amino acid-based pLMs in tasks such as melting point prediction, solubility prediction, subcellular localization classification, and function prediction. However, this codon-level information has, until now, gone unused for the task of post translational modification (PLM) prediction. 

#### Wavelet Kolmogorov-Arnold Network (Wav-KAN)
Traditionally, post translational modification (PLM) prediction models have relied on multi-layer perceptrons (MLPs) as the prediction engine to output the classification of a residue. These consist of multiple layers of neurons or nodes which use fixed, non-linear activation functions in order to learn complex relationships in the input data. Inputs to each neuron/node also utilize trainable weights in order to strengthen connections which are important to prediction. Kolmogorov-Arnold networks (KANs) modify this approach by using trainable, non-linear activation functions along the edges of the network instead of at the nodes. The nodes then simply sum the inputs to produce an output. The paper which proposes KANs can be found here: [Kolmogorov-Arnold Network Paper](https://arxiv.org/abs/2404.19756) and the accompanying Github repository here: [pyKAN Github](https://github.com/KindXiaoming/pykan).

Furthermore, KANs can be modified to include wavelet functions. Wavelets allow efficient capturing of both low and high-frequency components of the data. When wavelets are included into a KAN, the peformance, efficiency, and robustness of the model improves. The in-depth and original proposition of a wavelet-KAN (Wav-KAN) can be found here: [Wavelet Kolmogorov-Arnold Network Paper](https://arxiv.org/abs/2405.12832) as well as the Github repository which implements it here: [Wav-KAN Github](https://github.com/zavareh1/Wav-KAN).

### Overview
CaLMPhosKAN is a general phosphorylation PTM predictor which leverages multiple new techniques in the field in order to achieve high performance. Codon-aware embeddings are combined with amino acid-aware embeddings to create a representation of the proteins sequence with high amounts of valuable information. Feature extraction is performed using a 2D-convolutional layer followed by a Bidirectional Gated Recurrent Unit (BiGRU) in order to find spacial and sequential relationships between residues within a windowed frame. These features can then be learned by the Wav-KAN module and produce the final prediction. Using this process, CaLMPhosKAN outperforms other models in phosphorylation prediction with multiple new techniques.

## Architecture

<p align="center">
<img src="images/Calmphoskan_architecture.png"/> 
</p>

## Independent Test Set Evaluation
### Install Libraries

Python Version: `3.11.7`

To install the required libraries, run the following command:

```shell
pip install -r requirements.txt
```
This will install these libraries and versions:

<code>numpy==2.0.1
pandas==2.2.2
pyfiglet==1.0.2
scikit_learn==1.5.1
tabulate==0.9.0
torch==2.3.1
torchvision==0.18.1
tqdm==4.66.4</code>

### Download Testing Data
The necessary independent testing data is availiable to download here: [CaLMPhosKAN Independent Test Data](https://drive.google.com/drive/folders/16GBz_CJCvvUyhspVAw4Qi6upQRqGRciS?usp=drive_link). This is a folder which consists of these four files:
|Name|ST_dataset.npy|ST_labels.csv|Y_dataset.npy|Y_labels.csv|
|----|--------------|-------------|-------------|------------|
|Size|6.07 GB|197 KB|1009 MB|32 KB|

Once the files are downloaded, add the data folder to the same directory as the evaluation script.

### Run Evaluation
After the data folder is in place and the requirements are installed, evaluation can be performed with format of the command below:
```shell
python evaluate.py data
```
Note that 'data' must be replaced by ST or Y, signaling to the program to predict phosphorylation on either the serine & threonine (ST) dataset, or the tyrosine (Y) dataset. 

