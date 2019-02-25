# VAT-pytorch

## Hyper-parameters tuning
We explore the range of epsilon by [2.5, 5] and use same ones for the others.
The number of labeled data we use is 100.

|    epsilon    |   2   |  2.5  |   3   |  3.5  |   4   |  4.5  |   5   |
|:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Test accuracy | 97.35 | 97.18 | 97.45 | 97.21 | 97.36 | 98.34 | 97.84 |

In the paper, the test accuracy is reported as 98.64 (+-0.03).
Maybe this is because of rough hyper-parameter tuning or subtle implementation differences.
If the implementation or experimental settings are wrong compared to the ones in original paper,
please let me know.

## TODO
- Improve performance.