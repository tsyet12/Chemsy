# Chemsy - A Minimalistic Automatic Framework for Chemometrics and Machine Learning

![Chemsy Logo](https://github.com/tsyet12/Chemsy/blob/d46d0f8c1ab0b372b4684937d478ca6deaeba341/misc/wlogo.jpg)


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Current support for algorithms](#current-support-for-algorithms)
* [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
  * [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


<!-- ABOUT THE PROJECT -->
## About The Project
This project is to make a lightweight and flexible automatic framework for chemometrics and machine learning. The main target for the methods are for spectroscopic data and industrial process data analysis. Chemsy provides a structured, customizable and minimalistic framework for automatic pre-processing search. The syntax of Chemsy also follows the widely-used sklearn library, and any algorithms/method that has the sklearn syntax will be usable in Chemsy.

## Current support for algorithms

Automatic pre-processing search with support for:
- Partial Least Squares with Cross Validation
- Savitzky–Golay filter
- Asymmetric Least Squares (AsLS) Baseline Correction
- Modified Polynomial Baseline Correction
- Improved Modified Polynomial Baseline Correction
- Zhang Fit Baseline Correction
- Linear Baseline Correction
- Second Order Baseline Correction
- Multiplicative Scatter Correction
- First Derivative
- Second Derivative
- Standard Normal Variate
- Robust Normal Variate
- Standard Scaler (also known as Autoscaling)
- Min Max Scaler
- Any other algorithms with sklearn syntax can be used directly

To see what are the most updates algorithms available:
```python
import chemsy
from chemsy.help import see_methods

# see what preprocessing methods are available
see_methods(chemsy.prep.methods)

# see what prediction methods are available
see_methods(chemsy.predict.methods)
```
Return:
```
Preprocessing method supported:
['BaselineASLS', 'BaselineIModPoly', 'BaselineLinear', 'BaselineModpoly', 'BaselineSecondOrder', 'BaselineZhangFit', 'FirstDerivative', 'FunctionTransformer', 'KernelPCA', 'MSC', 'MaxAbsScaler', 'MinMaxScaler', 'PCA', 'PowerTransformer', 'QuantileTransformer', 'RNV', 'RobustScaler', 'SNV', 'SavgolFilter', 'SecondDerivative', 'StandardScaler']

Prediction method supported:
['BayesianRidge', 'DecisionTreeRegressor', 'ElasticNet', 'GaussianProcessRegressor', 'GradientBoostingRegressor', 'KNeighborsRegressor', 'KernelRidge', 'Lasso', 'LinearRegression', 'MLPRegressor', 'PLSRegression', 'PartialLeastSquaresCV', 'RandomForestRegressor', 'Ridge']

```


<!-- GETTING STARTED -->
## Getting Started

### Quick evaluation on Google Colab:
For quickstart/evaluation of the functionality, see [`this colab`](https://colab.research.google.com/drive/19_nPiAOQN9o5kxnXBjYqDvgEGhbZGD2K?usp=sharing) notebook online.

### Quick functionality in 3 Steps:

1. Import libraries and load dataset
```python
# Import all modules necessary 
import chemsy
from chemsy.explore import SupervisedChemsy
from chemsy.prep.methods import *
from chemsy.predict.methods import *
import numpy as np
import pandas as pd

# Use a default dataset
from sklearn.datasets import load_diabetes
X, Y = load_diabetes(return_X_y=True)
```

2. Make a custom recipe
```
# Make a custom recipe for the method search, all combinations will be evaluated
custom_recipe= {
"Level 0":[None],
"Level 1":[MSC(),StandardScaler(),MinMaxScaler(),RobustScaler()],
"Level 2":[PowerTransformer(),QuantileTransformer(output_distribution='normal', random_state=0), PCA(n_components='mle')],
"Level 3":[PartialLeastSquaresCV(),Lasso(), ]
}
```

3. Search pre-processing methods 
```
# Search pre-processing methods and all combinations
solutions=SupervisedChemsy(X, Y,recipe=custom_recipe)

# Show the results
solutions.get_results(verbose=False)
```
Return:
| Methods                                                      |   fit_time |   score_time |   cross_val_MAE |   cross_val_MSE |   cross_val_R2 |   cross_val_MBE |
|:-------------------------------------------------------------|-----------:|-------------:|----------------:|----------------:|---------------:|----------------:|
| StandardScaler + PCA + PartialLeastSquaresCV                 | 0.177647   |  0.00294271  |         43.1078 |         2816.97 |      0.513709  |        0.72431  |
| MinMaxScaler + PCA + PartialLeastSquaresCV                   | 0.185936   |  0.00269322  |         43.2748 |         2852.44 |      0.50761   |        0.522684 |
| StandardScaler + PCA + Lasso                                 | 0.00312543 |  0.00111251  |         43.3569 |         2832.88 |      0.510979  |        0.908942 |
| RobustScaler + PCA + PartialLeastSquaresCV                   | 0.221452   |  0.00257006  |         43.3624 |         2832.27 |      0.51107   |        0.871943 |
| StandardScaler + PowerTransformer + PartialLeastSquaresCV    | 0.201116   |  0.00330443  |         43.8542 |         2883.86 |      0.502165  |        0.922369 |
| RobustScaler + PowerTransformer + PartialLeastSquaresCV      | 0.205665   |  0.00339861  |         43.8731 |         2885.57 |      0.501871  |        0.925456 |
| MinMaxScaler + PowerTransformer + PartialLeastSquaresCV      | 0.207717   |  0.00376263  |         43.8793 |         2886.62 |      0.501692  |        0.913203 |
| RobustScaler + PCA + Lasso                                   | 0.00443821 |  0.000947857 |         43.9422 |         2868.85 |      0.504742  |        1.22346  |
| StandardScaler + PowerTransformer + Lasso                    | 0.0244179  |  0.00203629  |         44.2158 |         2911.23 |      0.49731   |        1.27553  |
| RobustScaler + PowerTransformer + Lasso                      | 0.025683   |  0.00173001  |         44.2366 |         2913.06 |      0.496988  |        1.28165  |
| MinMaxScaler + PowerTransformer + Lasso                      | 0.0320802  |  0.00213027  |         44.2526 |         2915.29 |      0.496592  |        1.28109  |
| RobustScaler + QuantileTransformer + PartialLeastSquaresCV   | 0.209099   |  0.00946445  |         44.9962 |         2979.41 |      0.486111  |        0.985491 |
| MinMaxScaler + QuantileTransformer + PartialLeastSquaresCV   | 0.207131   |  0.00927935  |         44.9968 |         2979.54 |      0.486091  |        0.984019 |
| StandardScaler + QuantileTransformer + PartialLeastSquaresCV | 0.201937   |  0.00911026  |         44.9971 |         2979.49 |      0.486097  |        0.986858 |
| MinMaxScaler + PCA + Lasso                                   | 0.00237641 |  0.000942993 |         45.0336 |         3013.84 |      0.479011  |        1.28565  |
| RobustScaler + QuantileTransformer + Lasso                   | 0.0161179  |  0.00758591  |         45.2021 |         2989.94 |      0.483484  |        1.46429  |
| MinMaxScaler + QuantileTransformer + Lasso                   | 0.0134507  |  0.0075232   |         45.2027 |         2990.07 |      0.483464  |        1.46294  |
| StandardScaler + QuantileTransformer + Lasso                 | 0.0138698  |  0.00747972  |         45.2037 |         2990.1  |      0.483459  |        1.4657   |
| MSC + PowerTransformer + PartialLeastSquaresCV               | 0.389996   |  0.0579424   |         63.8723 |         5962.2  |     -0.0291604 |       -2.21014  |
| MSC + PowerTransformer + Lasso                               | 0.244706   |  0.0584606   |         64.7382 |         6155.79 |     -0.0650035 |       -3.12385  |
| MSC + QuantileTransformer + Lasso                            | 0.231418   |  0.0727204   |         65.0553 |         6172.64 |     -0.0667151 |       -0.783697 |
| MSC + PCA + PartialLeastSquaresCV                            | 0.314129   |  0.0510806   |         65.7188 |         6031.59 |     -0.0421737 |        3.18991  |
| MSC + PCA + Lasso                                            | 0.205683   |  0.0513979   |         65.7989 |         6071.98 |     -0.0493179 |        2.92994  |
| MSC + QuantileTransformer + PartialLeastSquaresCV            | 0.336759   |  0.0603037   |         66.0683 |         6172.09 |     -0.0666824 |        3.56341  |


## Installation

### Install on [`Google Colab`](https://colab.research.google.com):
In a Colab code block:
```python
!pip install git+https://github.com/tsyet12/Chemsy --quiet
```


### Install on local python environment:
In a environment terminal or CMD:
```bat
pip install git+https://github.com/tsyet12/Chemsy --quiet
```


<!-- USAGE EXAMPLES -->
## Usage

To be updated.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b testbranch/prep`)
3. Commit your Changes (`git commit -m 'Improve testbranch/prep'`)
4. Push to the Branch (`git push origin testbranch/prep`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the Open Sourced BSD-2-Clause License. See [`LICENSE`](https://github.com/tsyet12/Chemsy/blob/main/LICENSE) for more information.



<!-- CONTACT -->
## Contact

Sin Yong Teng: sinyong.teng@ru.nl or tsyet12@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
Martijn Dingemans martijn.dingemans@gmail.com
