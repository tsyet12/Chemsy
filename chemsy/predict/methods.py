from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from sklearn.model_selection import ShuffleSplit

from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge, BayesianRidge
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


class PartialLeastSquaresCV(BaseEstimator):
  def __init__(self,cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=999),scoring='neg_mean_absolute_error',epsilon=0.0001,max_lv=30):
      self.__name__='PartialLeastSquaresCV'
      self.cv=cv
      self.model=None
      self.scoring=scoring
      self.epsilon=epsilon
      self.max_lv=max_lv
  def predict(self, X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return self.model.predict(X)
    
  def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
        y=pd.DataFrame(y)
      except:
        pass
      try:
        y=y.to_frame()
      except:
        pass
      max=min(X.shape[0], X.shape[1], y.shape[0],self.max_lv)
      hist_score=-np.inf
      flag=False
      for i in range(1,max+1):
        if flag==False:
          model=PLSRegression(n_components=i,scale=False)
          model.fit(X,y)
          cv_score=cross_validate(model, X, y, cv=self.cv,scoring=self.scoring)['test_score'].mean()
          #print(cross_validate(model, X, y, cv=self.cv,scoring=self.scoring)['test_score'].mean())
          #print(cv_score)
          #print(hist_score+self.epsilon)
          if cv_score>hist_score+self.epsilon:
            self.model=model
            self.__name__='PLS (n = '+str(i)+')'
            hist_score=cv_score
          else:
            flag=True
      return self


if __name__=='__main__':
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    plscv=PartialLeastSquaresCV()
    plscv.fit(X,y)
    plscv.predict(X)