# A Plug-and-Play Adapter for Task-Oriented Dialogue Systems


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

The script **train.py** acts as a main function to the project, you can run the experiments by the following commands:

```
python -u train.py --cfg ppa/ppa_BERT.cfg
```
Due to some stochastic factors(*e.g*., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. All the hyper-parameters are in the `configure/ppa/ppa_BERT.py` 

## Acknowledgement

This code is based on the released code (https://github.com/yizhen20133868/CI-ToD) for "Don’t be Contradicted with Anything!CI-ToD: Towards Benchmarking Consistency for Task-oriented Dialogue System" ***EMNLP2021***.[[PDF]](https://arxiv.org/pdf/2109.11292.pdf) .

For the pre-trained language model, we use huggingface's Transformer (https://huggingface.co/transformers).

We are grateful for their excellent works.
