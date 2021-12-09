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
This project is to make a lightweight and flexible automatic framework for chemometrics and machine learning. The main target for the methods are for spectroscopic data and industrial process data analysis. Chemsy provides a structured, customizable and minimalistic framework for automatic pre-processing search. The syntax of Chemsy also follows the widely-used sklearn library, and any algorithms/method that has the sklearn syntax will be usable in Chemsy. Chemsy supports freedom, open source and software accessability for all chemometricians, machine learning engineers and data scientists.

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
For quickstart/evaluation of the functionality, see **[`this Google Colab`](https://colab.research.google.com/drive/19_nPiAOQN9o5kxnXBjYqDvgEGhbZGD2K?usp=sharing)** notebook online.

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
```python
# Make a custom recipe for the method search, all combinations will be evaluated
custom_recipe= {
"Level 0":[None],
"Level 1":[MSC(),StandardScaler(),MinMaxScaler(),RobustScaler()],
"Level 2":[PowerTransformer(),QuantileTransformer(output_distribution='normal', random_state=0), PCA(n_components='mle')],
"Level 3":[PartialLeastSquaresCV(),Lasso(), ]
}
```

3. Search pre-processing methods 
```python
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
| ⋮  | ⋮   |   ⋮ |         ⋮ |         ⋮ |      ⋮  |        ⋮ |


## Installation

### Install on Google Colab:
In a [`Colab`](https://colab.research.google.com) code block:
```bat
!pip install git+https://github.com/tsyet12/Chemsy --quiet
```


### Install on local python environment:
In a environment terminal or CMD:
```bat
pip install git+https://github.com/tsyet12/Chemsy --quiet
```


<!-- USAGE EXAMPLES -->
## Usage

A recipe from Engel et al. (2013) for spectroscopic IR data:
```python
Engel_2013= {
"Baseline":[None, BaselineSecondOrder(),BaselineSecondOrder(degree=3),BaselineSecondOrder(degree=4),BaselineASLS(),FirstDerivative(),SecondDerivative()],
"Scatter":[None, MeanScaling(), MedianScaling(),MaxScaling(),L2NormScaling(),RNV(q=0.15),RNV(q=0.25),RNV(q=0.35),MSC()],
"Noise":[None, SavgolFilter(5,2),SavgolFilter(9,2),SavgolFilter(11,2),SavgolFilter(5,3),SavgolFilter(9,3),SavgolFilter(11,3),SavgolFilter(5,4),SavgolFilter(9,4),SavgolFilter(11,4)],
"Scaling & Transformations":[MeanCentering(),StandardScaler(),RangeScaling(),ParetoScaling,PoissonScaling(),LevelScaling(), ],
"PLS":[PartialLeastSquaresCV()]
}

```
Recipe reference:
Engel, J., Gerretzen, J., Szymańska, E., Jansen, J.J., Downey, G., Blanchet, L. and Buydens, L.M., 2013. Breaking with trends in pre-processing?. TrAC Trends in Analytical Chemistry, 50, pp.96-106.https://www.sciencedirect.com/science/article/pii/S0165993613001465


More to be updated.


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
