# Description 

This is an unofficical implementation about the paper "[neuro2vec: Masked Fourier Spectrum Prediction for Neurophysiological Representation Learning](https://arxiv.org/abs/2204.12440)". Mainly focus on sleep stage classification task.

## Include

- MAE implementation on neurophysiological signal
- ViT concept on neurophysiological signal
- neuro2vec implementation

## Implementation detail

> terms which are not declare in the paper

- the heads of MHA -> `4`
- the pooling method of Transformer -> `cls token`
- the position embedding -> `Sinusoidal`
- the implementation of heads -> `Linear`
- the initialization of pos_embed and mask_token -> `zero`

## Experimental results

> using the model of last epoch for evaluation

- supervised for transformer
  - `Exp` Acc -> 80.96, F1 -> 74.23
  - `Paper` Acc -> 84.20 F1 -> 77.15