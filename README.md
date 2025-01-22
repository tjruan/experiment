**This work assesses the privacy risk of smart meters through directed information from grid loads to sensitive home operating states to model the causality of privacy breaches. The VCA-PPO approach is proposed by jointly utilizing the ideas of DRL, Secure RL and variational methods to effectively address privacy preserving cost effective EMU policy design.**

## Prerequisites

Python version : ` >= Python 3.8 `

Overall, the required python packages are listed as follows:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [torch==2.4.1](https://pytorch.org/docs/2.4.1/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)

## Installation

Use pip to install all the required libraries listed in the requirements.txt file.

```
pip install -r requirements.txt
```

## Data generation

The dataset used in this article was generated based on a transfer probability matrix：

- We set each time slot to `2` minutes, a time block consists of `T = 720` time slots, corresponding to `24` hours. We consider `n = 5` different house appliances.

You may find more information on how to generate the datasets in the [data]([experiment/data at main · tjruan/experiment](https://github.com/tjruan/experiment/tree/main/data)) folder.

## Train

You can train by running the `train_PPOLag.py` file, in which you can set the relevant parameters for the experiment:

- Note that the random number seed may affect the results.
- The model results obtained from the experiment will be saved in the `run` folder.

You can change the parameters of the experiment by modifying the contents of the `train_PPOLag.py` file.

## Test & Result

The `test` folder contains a sample to test the results of the experiment, which will be saved in the `result` folder.



