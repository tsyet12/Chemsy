from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from sklearn.model_selection import ShuffleSplit

class PartialLeastSquares(BaseEstimator):
  def __init__(self,cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=999)):
      self.__name__='PLS_CV'
      self.cv=cv
      self.model=None
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
      max=min(X.shape[0], X.shape[1], y.shape[1])
      hist_score=-np.inf
      flag=False
      for i in range(1,max+1):
        if flag==False:
          model=PLSRegression(n_components=i,scale=False)
          model.fit(X,y)
          cv_score=cross_validate(model, X, y, cv=self.cv,scoring='r2')['test_score'].mean()
          if cv_score>hist_score:
            self.model=model
            self.__name__='PLS (n = '+str(i)+')'
            hist_score=cv_score
          else:
            flag=True
      return self
