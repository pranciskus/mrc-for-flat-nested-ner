# A Unified MRC Framework for Named Entity Recognition 
The repository contains the code of the recent research advance in [Shannon.AI](http://www.shannonai.com). 

**A Unified MRC Framework for Named Entity Recognition** <br>
Preprint. [arXiv](https://arxiv.org/abs/1910.11476)<br>
**Authors:** Xiaoya Li\*, Jingrong Feng\*, Yuxian Meng, Qinghong Han, Fei Wu, Jiwei Li<br>
**BibTex:** 
```latex
@article{li2019unified,
  title={A Unified MRC Framework for Named Entity Recognition},
  author={Li, Xiaoya and Feng, Jingrong and Meng, Yuxian and Han, Qinghong and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1910.11476},
  year={2019}
}
```
For any question, please feel free to contact xiaoya_li@shannonai.com or post Github issue.<br>

![docs/overview.png](./docs/overview.png)

## Overview 

## Contents
1. [Experimental Results](#experimental-results)
  * [Flat NER](##flat-ner) 
  * [Nested NER](##nested-ner)
2. [Datasets Preparation](#datasets-preparation)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
5. [Updates](#updates)
6. [FAQ](#faq)

## Experimental Results  
Experiments are conducted both on *Flat* and *Nested* NER datasets.
We only list comparisons of our proposed method with previous SOTA in terms of span-level micro-averaged F1-score here.  
More comparisons and span-level micro Precision/Recall scores could be found in the [paper](https://arxiv.org/abs/1910.11476.pdf). 
### Flat NER 
Evaluations are conducted on the widely-used bechmarks: `CoNLL2003`, `OntoNotes 5.0` for English; `MSRA`, `OntoNotes 4.0` for Chinese. 

| Dataset | Eng-CoNLL03 | Eng-OntoNotes5.0 | Zh-MSRA | Zh-OntoNotes4.0 | 
|---|---|---|---|---|
| Previous SOTA | 92.8 | 89.16 | 95.54  | 80.62 | 
| Our method | **93.04** | **91.11** | **95.75** | **82.11** | 
|  | **(+0.24)** | **(+1.95)** | **(+0.21)** | **(+1.49)** | 

* refers 

### Nested NER
Evaluations are conducted on the widely-used `ACE 2004`, `ACE 2005`, `GENIA`, `KBP-2017` English datasets.

| Dataset | ACE 2004 | ACE 2005 | GENIA | KBP-2017 | 
|---|---|---|---|---|
| Previous SOTA | 84.7 | 84.33 | 78.31  | 74.6 | 
| Our method | **85.98** | **86.88** | **83.75** | **80.97** | 
|  | **(+1.28)** | **(+2.55)** | **(+5.44)** | **(+6.37)** | 

## Datasets Preparation

We release preprocessed and source datasets for both flat and nested NER benchmarks. <br>

You can download the preprocessed datasets from [Google Drive](./docs/preprocess_data.md). The source data files could be found at [Google Drive](./docs/source_data.md)

For data processing, you can follow the [guidance](./docs/data_preprocess.md) to generate your own MRC-based entity recognition training files. 

## Depndencies 

## Usage 
 
## Updates 
 
11/13/2019:
    1. init commit 

## FAQ