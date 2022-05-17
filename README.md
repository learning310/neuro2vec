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
- the initialization of mask_token -> `zero`

## Experimental results

> using the model of last epoch for evaluation

- supervised for transformer
  - `Exp` Acc -> 82.78, F1 -> 76.31
  - `Paper(Worst)` Acc -> 83.80 F1 -> 76.55

## Experience

1. PreNorm better than PostNorm