# A Plug-and-Play Adapter for Task-Oriented Dialogue Systems

## Abstract

Task-oriented Dialogue system (ToD) has gained significant attention due to its aim to assist users in accomplishing various tasks. However, the neural network-based dialogue system is like a black box, which may lead to erroneous responses and result in an unfriendly user experience. To address this issue, consistency identification is proposed to prevent generating inconsistent responses. However, the existing consistency identification methods require frequent interaction with the knowledge base, making them susceptible to the introduction of noise during the knowledge base fusion process, ultimately leading to a decline in performance. In this paper, we propose a plug-and-play method for consistency identification, which can introduce external knowledge into the internal reasoning process of the pre-trained language model (PLM) without modifying PLM’s structure. Additionally, we design a new fusion mechanism that effectively fuses the knowledge base information related to the current utterance, which helps the model avoid introducing noise from the irrelevant knowledge base. The experimental results demonstrate that our method achieves state-of-the-art performance on the consistency identification task, improving F1 scores by 2.9% absolute points over the previous methods. Finally, we investigate different knowledge base fusion methods and provide extensive experiments to show the advantages of our proposed method.

## Preparation

The  packages we used are listed follow:

```
-- scikit-learn==0.23.2
-- numpy=1.19.1
-- pytorch>=1.1.0
-- fitlog==0.9.13
-- tqdm=4.49.0
-- sklearn==0.0
```

## How to Run it

Before running the code, please unzip the **transformers.zip** file. The model's Adapter module and fusion layer are in **modeling_bert.py**

The script **train.py** acts as a main function to the project, you can run the experiments by the following commands:

```
python -u train.py --cfg ppa/ppa_BERT.cfg
```
Due to some stochastic factors(*e.g*., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. All the hyper-parameters are in the `configure/ppa/ppa_BERT.py` 

## Citation

If you use any source codes or the datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

```
@article{DING2024103637,
title = {A plug-and-play adapter for consistency identification in task-oriented dialogue systems},
journal = {Information Processing & Management},
volume = {61},
number = {3},
pages = {103637},
year = {2024},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2023.103637},
url = {https://www.sciencedirect.com/science/article/pii/S0306457323003746},
author = {Zeyuan Ding and Zhihao Yang and Hongfei Lin},
keywords = {Consistency identification, Task-oriented dialogue, Knowledge injection, Fusion mechanism, Adapter module},
abstract = {Task-oriented Dialogue system (ToD) has gained significant attention due to its aim to assist users in accomplishing various tasks. However, the neural network-based dialogue system is like a black box, which may lead to erroneous responses and result in an unfriendly user experience. To address this issue, consistency identification is proposed to prevent generating inconsistent responses. However, the existing consistency identification methods require frequent interaction with the knowledge base, making them susceptible to the introduction of noise during the knowledge base fusion process, ultimately leading to a decline in performance. In this paper, we propose a plug-and-play method for consistency identification, which can introduce external knowledge into the internal reasoning process of the pre-trained language model (PLM) without modifying PLM’s structure. Additionally, we design a new fusion mechanism that effectively fuses the knowledge base information related to the current utterance, which helps the model avoid introducing noise from the irrelevant knowledge base. The experimental results demonstrate that our method achieves state-of-the-art performance on the consistency identification task, improving F1 scores by 2.9% absolute points over the previous methods. Finally, we investigate different knowledge base fusion methods and provide extensive experiments to show the advantages of our proposed method.}
}
```

## Acknowledgement

This code is based on the released code (https://github.com/yizhen20133868/CI-ToD) for "Don’t be Contradicted with Anything!CI-ToD: Towards Benchmarking Consistency for Task-oriented Dialogue System" https://arxiv.org/pdf/2109.11292.pdf).

For the pre-trained language model, we use huggingface's Transformer (https://huggingface.co/transformers).

We are grateful for their excellent works.
