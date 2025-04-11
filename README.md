# Learning Efficient Positional Encodings with GNNs
By Charilaos Kanatsoulis, Evelyn Choi, Stefanie Jegelka, Jure Leskovec, Alejandro Ribeiro.

![PE](https://github.com/ehejin/PEARL/blob/main/PE_final.png)

## Overview
This repository contains the official implementation and experiment code for the paper, **"Learning Efficient Positional Encodings with GNNs"**. 

### Experiments
* Instructions and experiments for PEARL on the ZINC, DrugOOD, and Peptides datasets can be found in PEARL/configs.
* Experiments for PEARL on the REDDIT datasets can be found in Pearl-Reddit.
* Experiments for PEARL on the RelBench dataset can be found in the relbench folder.

### Dependencies and Attribution
This project builds upon code from the SignNet, SPE, and RelBench repositories. We have modified and extended components from these frameworks for our use.
We use the [[SignNet repo](https://github.com/cptq/SignNet-BasisNet)] by Lim et al., 2022, the [[SPE repo](https://github.com/Graph-COM/SPE)] by Huang et al., 2024, and the [[RelBench repo](https://github.com/snap-stanford/relbench)] by Robinson et al., 2024.