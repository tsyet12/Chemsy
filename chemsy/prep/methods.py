
#dependencies
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from scipy import sparse, signal
from BaselineRemoval import BaselineRemoval
from sklearn.model_selection import ShuffleSplit
from scipy.sparse.linalg import spsolve

#Import prep methods
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA
   


class SavgolFilter(TransformerMixin):
  def __init__(self,window_length=5,polyorder=2,axis=1,  *args, **kwargs):
      self.__name__='SavgolFilter'
      self.window_length=window_length
      self.polyorder=polyorder
      self.axis=axis
      self.output=None
  def fit(self,X,y=None):
      pass

  def transform(self,X,y=None):
      self.output=savgol_filter(X,window_length=self.window_length,polyorder=self.polyorder,axis=self.axis)
      return self.output

  def fit_transform(self,X,y=None):
      self.output=savgol_filter(X,window_length=self.window_length,polyorder=self.polyorder,axis=self.axis)
      return self.output

class BaselineASLS(TransformerMixin):
  #Asymmetric Least Squares
  def __init__(self, lam=1e5, p=1e-3, niter=10):
      self.__name__='BaselineAsLS'
      self.lam=lam
      self.p=p
      self.niter=niter
      self.y=None
      self.output=None
  def fit(self,X,y):
      self.y=y
  def transform(self,X,y=None):
      y=self.y
      self.output=np.apply_along_axis(lambda x: self.line_remove(x), 0, X)        
      return self.output
  def line_remove(self,f):
      L = len(f)
      D = sparse.csc_matrix(np.diff(np.eye(L), 2))
      w = np.ones(L)
      z = 0
      for i in range(self.niter):
          W = sparse.spdiags(w, 0, L, L)
          Z = W + self.lam * D.dot(D.transpose())
          z = spsolve(Z, w * f)
          w = self.p * (f > z) + (1 - self.p) * (f < z)
      return z    
  def fit_transform(self,X,y=None):
      self.y=y
      return self.transform(X,y)



class BaselineModpoly(BaseEstimator,TransformerMixin):
  def __init__(self, degree=2):
    self.__name__='BaselineModPoly'
    self.degree=degree
  def fit(self,X,y):
    pass
  def transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    X_=np.zeros_like(X)
    for i in range(X.shape[0]):
      MP=BaselineRemoval(X[i,:])
      X_[i,:]=MP.ModPoly(self.degree)
      del MP
    return X_
  def fit_transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    X_=np.zeros_like(X)
    for i in range(X.shape[0]):
      MP=BaselineRemoval(X[i,:])
      X_[i,:]=MP.ModPoly(self.degree)
      del MP
    return X_


class BaselineZhangFit(BaseEstimator,TransformerMixin):
  def __init__(self, itermax=50):
    self.__name__='BaselineZhangFit'
    self.itermax=itermax
  def fit(self,X,y):
    pass
  def transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    X_=np.zeros_like(X)
    for i in range(X.shape[0]):
      MP=BaselineRemoval(X[i,:])
      X_[i,:]=MP.ZhangFit(itermax=self.itermax)
      del MP
    return X_
  def fit_transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    X_=np.zeros_like(X)
    for i in range(X.shape[0]):
      MP=BaselineRemoval(X[i,:])
      X_[i,:]=MP.ZhangFit(itermax=self.itermax)
      del MP
    return X_

class BaselineIModPoly(BaseEstimator,TransformerMixin):
  def __init__(self, degree=2):
    self.__name__='BaselineImprovedModPoly'
    self.degree=degree
  def fit(self,X,y):
    pass
  def transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    X_=np.zeros_like(X)
    for i in range(X.shape[0]):
      MP=BaselineRemoval(X[i,:])
      X_[i,:]=MP.IModPoly(self.degree)
      del MP
    return X_
  def fit_transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    X_=np.zeros_like(X)
    for i in range(X.shape[0]):
      MP=BaselineRemoval(X[i,:])
      X_[i,:]=MP.IModPoly(self.degree)
      del MP
    return X_

class BaselineLinear(BaseEstimator,TransformerMixin):
  def __init__(self):
    self.__name__='BaselineLinear'
  def fit(self,X,y):
    pass
  def transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    return signal.detrend(X)
  def fit_transform(self,X,y=None):
    try:
      X=X.to_numpy()
    except:
      pass
    return signal.detrend(X)


class BaselineSecondOrder(BaseEstimator,TransformerMixin):
  def __init__(self,degree=2):
      self.__name__='BaselineSecondOrder'
      self.degree=degree
  def fit(self,X,y):
      pass
  def fit_transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      t=np.arange(0,X.shape[1])
      X_s= X.apply(lambda x: x- np.polyval(np.polyfit(t,x,self.degree), t),axis=1)
      return  X_s
  def transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      t=np.arange(0,X.shape[1])
      X_s= X.apply(lambda x: x- np.polyval(np.polyfit(t,x,self.degree), t),axis=1)
      return  X_s


class MSC(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.__name__='MSC'
    def fit(self,X,y):
        pass
    def transform(self,X,y=None):
        try:
          X=pd.DataFrame(X)
        except:
          pass
        mean= np.array(X.mean(axis=0))
        def transformMSC(x,mean):
            m,b= np.polyfit(mean,x,1)
            return (x-b)*m
        return X.apply(transformMSC,args=(mean,),axis=1).values

    def fit_transform(self,X,y=None):
        try:
          X=pd.DataFrame(X)
        except:
          pass
        self.mean= np.array(X.mean(axis=0))
        def transformMSC(x,mean):
            m,b= np.polyfit(mean,x,1)
            return (x-b)*m
        return X.apply(transformMSC,args=(self.mean,),axis=1).values


class FirstDerivative(BaseEstimator,TransformerMixin):
    def __init__(self,d=2):
        self.__name__='First Derivative'
        self.d=d
    def fit(self,X,y):
        pass
    def transform(self,X,y=None): 
        try:
          X=pd.DataFrame(X)
        except:
          pass
        X_=X.diff(self.d,axis=1)
        drop= list(X_.columns)[0:2]
        X_.drop(columns=drop,inplace=True)    
        return X_
    def fit_transform(self,X,y=None):
        try:
          X=pd.DataFrame(X)
        except:
          pass
        X_=X.diff(self.d,axis=1)
        drop= list(X_.columns)[0:2]
        X_.drop(columns=drop,inplace=True)    
        return X_

# TO DO:
#Piecewise MSC (PMSC)
#Extended MSC (2nd order), Inverse MSC, EIMSC
#Weighted MSC, Loopy MSC (LMSC)
#Norris-Williams
#WhittakerSmooth

class SecondDerivative(BaseEstimator,TransformerMixin):
    def __init__(self,d=2):
        self.__name__='Second Derivative'
        self.d=d
    def fit(self,X,y):
        pass
    def transform(self,X,y=None): 
        try:
          X=pd.DataFrame(X)
        except:
          pass
        X_=X.diff(self.d,axis=1)
        drop= list(X_.columns)[0:2]
        X_.drop(columns=drop,inplace=True)  
        X_=X_.diff(self.d,axis=1) #second dev
        drop= list(X_.columns)[0:2]
        X_.drop(columns=drop,inplace=True) 
        return X_
    def fit_transform(self,X,y=None):
        try:
          X=pd.DataFrame(X)
        except:
          pass
        X_=X.diff(self.d,axis=1)
        drop= list(X_.columns)[0:2]
        X_.drop(columns=drop,inplace=True)  
        X_=X_.diff(self.d,axis=1) #second dev
        drop= list(X_.columns)[0:2]
        X_.drop(columns=drop,inplace=True)         
        return X_


class SNV(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='SNV'
    def fit(self,spc):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      X=X.T
      R=(X -X.mean(axis=0))/(X.std(axis=0)+0.0000001)
      return R.T
    def fit_transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      X=X.T
      R=(X -X.mean(axis=0))/(X.std(axis=0)+0.0000001)
      return R.T
class RNV(BaseEstimator,TransformerMixin):
    def __init__(self,q=0.1):
      self.__name__='RNV'
      self.q=q
    def fit(self,spc):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return (X -X.quantile(q=self.q,axis=1))/(X.quantile(q=self.q,axis=1).std(axis=1)+0.0000001)
    def fit_transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return (X -X.quantile(q=self.q,axis=1))/(X.quantile(q=self.q,axis=1).std(axis=1)+0.0000001)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path=r'C:\Users\User\Downloads\\'
    data=pd.read_excel(path+'Data1.xlsx',index_col=0)
    data=data.iloc[:100,:140]
    data.plot(legend=False)
    msc=SNV()
    trans_data=msc.fit_transform(data)
    trans_data=pd.DataFrame(trans_data)
    trans_data.plot(legend=False)
    plt.show()

