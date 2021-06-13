### SwishNet for ASR

This repository contains a Tensorflow v1 implementation of the SwishNet architecture (paper: https://arxiv.org/abs/1812.00149) and its training and testing on the MUSAN (https://arxiv.org/abs/1510.08484) and GTZAN  datasets, as well as the pre-processing of these datasets as described in the SwishNet paper. The raw datasets are not included, these are open access and can be downloaded from sites maintained by their authors. The main task is speech, music and noise detection.

The achieved results on MUSAN were: 

| Clip Length | Validation Overall  | Validation SNS | Test Overall | Test SNS |
| :---:   | :-: | :-: | :-: | :-: |
| 0.5s | 97.68% | 99.49% | 97.68% | 99.51% |
| 1.0s | 98.65% | 99.71% | 98.53% | 99.72% |
| 2.0s | 98.86% | 99.9% | 98.74% | 99.91% |

These results are comparable to Table II of the SwishNet paper (undistilled, wide model version) and roughly reach the same performance. Finetuning on GTZAN yielded an accuracy of 97.58% compared to the 98.00% achieved by the authors.
