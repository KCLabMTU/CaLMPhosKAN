<div align="center">
  
# CaLMPhosKAN

</div>


<p align="center">
  Predict General Phosphorylation Modification in Proteins Using the Fusion of Codon-Aware Embeddings and Amino Acid-Aware Embeddings with Wavelet-based Kolmogorov–Arnold Network
</p>

<p align="center">
<img src="images/example_run.png"/> 
</p>

<p align="center">
<a href="https://pypi.org/project/tabulate/"><img alt="tabulate" src="https://img.shields.io/badge/tabulate-0.9.0-blue.svg"/></a>  
<a href="https://pytorch.org/vision/stable/index.html"><img alt="torchvision" src="https://img.shields.io/badge/torchvision-0.18.1-red.svg"/></a>  
<a href="https://pypi.org/project/pyfiglet/"><img alt="pyfiglet" src="https://img.shields.io/badge/pyfiglet-1.0.2-yellow.svg"/></a>
<a href="https://www.python.org/"><img alt="python" src="https://img.shields.io/badge/Python-3.11.7-blue.svg"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3.1-orange.svg"/></a>
<a href="https://scikit-learn.org/"><img alt="scikit_learn" src="https://img.shields.io/badge/scikit_learn-1.5.1-blue.svg"/></a>
<a href="https://numpy.org/"><img alt="numpy" src="https://img.shields.io/badge/numpy-2.0.1-red.svg"/></a>
<a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-2.2.2-yellow.svg"/></a>
<a href="https://tqdm.github.io/"><img alt="tqdm" src="https://img.shields.io/badge/tqdm-4.66.4-blue.svg"/></a>
<a href="https://github.com/KCLabMTU/CaLMPhosKAN/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/KCLabMTU/CaLMPhosKAN.svg?style=flat&color=blue"></a>
<a href="https://github.com/KCLabMTU/CaLMPhosKAN/pulls"><img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/KCLabMTU/CaLMPhosKAN.svg?style=flat&color=blue"></a>
</p>

## About
The mapping from codon to amino acid is surjective due to the high degeneracy of the codon alphabet, suggesting that codon space might harbor higher information content. Embeddings from the codon language model have recently demonstrated success in various downstream tasks. However, predictive models for phosphorylation sites,arguably the most studied Post-Translational Modification (PTM), and PTM sites in general, have predominantly relied on amino acid-level representations. This work introduces a novel approach for the prediction of phosphorylation sites by incorporating codon-level information through embeddings from a recently developed codon language model trained on protein-coding DNA sequences. Protein sequences are first meticulously mapped to reliable coding sequences and encoded using this encoder to generate codon-aware embeddings. These embeddings are then integrated with amino acid-aware embeddings obtained from a protein language model through an early fusion strategy. Subsequently, a window-level representation of the site of interest, retaining full sequence context, is formed from the fused embeddings. A ConvBiGRU network extracts features capturing spatiotemporal correlations between proximal residues within the window, followed by a prediction head based on a Kolmogorov-Arnold Network (KAN) employing the Derivative of Gaussian (DoG) wavelet transform to produce the inference for the site.

#### Codon-aware Embeddings
The coding DNA sequences are encoded using a specialized codon-aware protein language model called CaLM (Codon adaptation Language Model) (22). Built on the Evolutionary Sequence Modelling (ESM) framework, CaLM utilizes an architecture comprising 12 encoder layers (each with 12 attention heads) and a prediction head, amounting to 86 million parameters in total. This model undergoes pretraining using a masked language modeling denoising objective on a dataset of approximately 9 million non-redundant coding sequences derived from whole-genome sequencing.

#### Amino Acid-aware Embeddings 
Amino acid-aware embeddings are derived from a protein language model trained on a large corpus of protein sequences. In this work, we utilize a ProtTrans family model called ProtT5, a prominent pLM established for its high performance in various protein downstream tasks (5), including post-translational modification prediction (25; 24; 23). ProtT5 is built on the T5 (Text-to-Text Transfer Transformer) architecture and has been trained using an MLM denoising objective on the UniRef50 (UniProt Reference Clusters, encompassing 45 million protein sequences) database. The model comprises a 24-layer encoder-decoder architecture (each with 32 attention heads) and contains approximately 2.8 billion learnable parameters.

#### Wavelet Kolmogorov-Arnold Network (Wav-KAN)
Traditionally, prediction models have relied on multi-layer perceptrons (MLPs) for classification tasks. MLPs consist of multiple layers of neurons or nodes, which use fixed non-linear activation functions to learn complex relationships in the input data. Each neuron/node utilizes trainable weights to strengthen the connections that are most important for prediction.

Kolmogorov–Arnold Networks (KANs) modify this approach by placing trainable, non-linear activation functions on the edges of the network, rather than at the nodes. The nodes themselves simply sum the inputs to produce an output. You can find the original paper proposing KANs here: Kolmogorov–Arnold Network Paper, along with the accompanying GitHub repository here: [pyKAN Github](https://github.com/KindXiaoming/pykan).

Additionally, KANs can be enhanced by incorporating wavelet functions, which allow efficient capture of both low- and high-frequency components in the data. When wavelets are integrated into a KAN, the model's performance, efficiency, and robustness improve. You can find the detailed proposition of the Wavelet-KAN (Wav-KAN) in this paper: [Wavelet Kolmogorov-Arnold Network Paper](https://arxiv.org/abs/2405.12832) and the corresponding GitHub repository here: [Wav-KAN Github](https://github.com/zavareh1/Wav-KAN).


## Architecture

<p align="center">
<img src="images/Calmphoskan_architecture.png"/> 
</p>

## Use this Repository
To start using this repository and obtain a local copy, you may clone it or download it directly from Github.

### Clone the Repository
By using Git (must be installed on your local system), you can clone the repository directly from your terminal or command line. The command to do so is here:

```shell
git clone git@github.com:KCLabMTU/CaLMPhosKAN.git
```

### Download the Repository Directly
If you choose not to use Git, you can still download the repository as a zip file by clicking the green '<>Code' dropdown box, selecting the 'Local' tab, and clicking 'Download ZIP' from the main CaLMPhosKAN page, or simply use this link: [Download main.zip](https://github.com/KCLabMTU/CaLMPhosKAN/archive/refs/heads/main.zip)

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
The necessary independent testing data is available to download here: [CaLMPhosKAN Independent Test Data](https://drive.google.com/drive/folders/16GBz_CJCvvUyhspVAw4Qi6upQRqGRciS?usp=drive_link). This is a folder which consists of these four files:
|Name|ST_dataset.npy|ST_labels.csv|Y_dataset.npy|Y_labels.csv|
|----|--------------|-------------|-------------|------------|
|Size|6.07 GB|197 KB|1009 MB|32 KB|

If you downloaded the entire 'data' folder, unzip the data folder into the same directory as the evaluation script inside the repository folder.
> <kbd>**Note:**</kbd>
> 1. Downloading the data in this way will cause only three of the four required files to be downloaded inside of a folder. 'ST_dataset.npy' is too large to be downloaded into a folder with the other files and must be added manually. You may unzip the downloaded folder with the three files to the previously described destination, then add 'ST_dataset.npy' to this folder.
> 2. It is likely that this fourth large file will not be named exactly 'ST_dataset.npy'. Before running the evaluation script, please make sure that the data folder is named 'data' and that the files contained within are named exactly as the above table describes.

If you downloaded each file independently, add each file to a new folder titled 'data', and add this to the same directory as the evaluation script inside the repository folder.
> <kbd>**Note:**</kbd>
Before running the evaluation script, please make sure that the data folder is named 'data' and that the files contained within are named exactly as the above table describes.

### Run Evaluation
After the data folder is in place and the requirements are installed, evaluation can be performed with the format of the command below:
```shell
python evaluate.py <dataset>
```
Note that `<dataset>' must be replaced by ST or Y, depending on which residue-specific model you want to run.

#### Output Examples
The output of a successful execution should look very similar to the screenshots that follow. Directly below is the example of the evaluation script which executed on the ST dataset...

<p align="center">
<img src="images/example_output_ST.png"/> 
</p>

...and below this is the example of the evaluation script which executed on the Y dataset.

<p align="center">
<img src="images/example_output_Y.png"/> 
</p>
