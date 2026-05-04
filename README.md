# Transfer Learning assessement for bulk gene expression data
Repository for the article: Reusing Biological Representations: Transfer Learning Assessment for bulk RNA-seq data

## Dataset
### The Cancer Genome Atlas (TCGA)
The Cancer Genome Atlas ([[TCGA]](https://portal.gdc.cancer.gov/)) collected many types of data for each of over 20,000 tumor and normal samples. Each step in the Genome Characterization Pipeline generated numer>
* clinical information (e.g., smoking status)
* molecular analyte metadata (e.g., sample portion weight)
* molecular characterization data (e.g., gene expression values)

## Installation
```
git clone git@github.com:kdradjat/transferlearning_rnaseq.git
cd transferlearning_rnaseq
python3 -m venv venv
source venv/bin/activate
pip install .
```
## Usage 
All the scripts used for the experiments with the differents methods are available on the 
