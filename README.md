<div align="center">
  
# CaLMPhosKAN

</div>


<p align="center">
  Predict General Phosphorylation Modification in Proteins using the Fusion of Codon-Aware Embeddings and Amino Acid-Aware Embeddings with Wavelet-based Kolmogorov–Arnold Network
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

