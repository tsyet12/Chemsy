from sklearn.datasets import load_diabetes
from chemsy.explore import SupervisedChemsy
from chemsy.prep.methods import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge, BayesianRidge
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA, KernelPCA
if __name__=="__main__":
    from sklearn.datasets import load_diabetes
    X, Y = load_diabetes(return_X_y=True)
    custom_recipe= {
    "Level 1":[StandardScaler(),MinMaxScaler(),RobustScaler()],
    "Level 2":[PowerTransformer(),QuantileTransformer(output_distribution='normal', random_state=0), PCA(n_components='mle')],
    "Level 3":[PartialLeastSquares()]
    }
    solutions=SupervisedChemsy(X, Y,recipe=custom_recipe)
    solutions.get_results()