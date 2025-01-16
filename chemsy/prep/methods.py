
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
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA
   


class SavgolFilter(BaseEstimator,TransformerMixin):
  def __init__(self,window_length=5,polyorder=2,axis=1):
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
      
  

class BaselineASLS(BaseEstimator,TransformerMixin):
  #Asymmetric Least Squares
  def __init__(self, lam=1e5, p=1e-3, niter=10):
      self.__name__='BaselineAsLS'
      self.lam=lam
      self.p=p
      self.niter=niter
      self.y=None
      self.output=None
 
  def fit(self,X,y=None):
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
          Z=W + self.lam * D.dot(D.transpose())
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
  def fit(self,X,y=None):
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

  def fit(self,X,y=None):
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

  def fit(self,X,y=None):
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

  def fit(self,X,y=None):
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
  
  def fit(self,X,y=None):
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
        self.mean=None
 
    def fit(self,X,y=None):
        self.mean= np.array(X.mean(axis=0))
    def transform(self,X,y=None):
        try:
          X=pd.DataFrame(X)
        except:
          pass
        #self.mean= np.array(X.mean(axis=0))
        def transformMSC(x,mean):
            m,b= np.polyfit(mean,x,1)
            return (x-b)*m
        return X.apply(transformMSC,args=(self.mean,),axis=1).values

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
    
    def fit(self,X,y=None):
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

    def fit(self,X,y=None):
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

    def fit(self,X):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
        if X.shape[1] == 1:
          X=X.T
      except:
        pass
      R=(X.subtract(X.mean(axis=1), axis=0)).divide(X.std(axis=1)+np.finfo(float).eps, axis=0)
      return R
    def fit_transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
        if X.shape[1] == 1:
          X=X.T
      except:
        pass
      self.fit(X)
      return self.transform(X)
       
class RNV(BaseEstimator,TransformerMixin):
    def __init__(self,q=0.5):
      self.__name__='RNV'
      self.q=q

    def fit(self,X):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
        if X.shape[1] == 1:
          X=X.T
      except:
        pass
      percentile=X.quantile(q=self.q,axis=1)
      percentile2=(np.asarray(percentile)*np.ones(X.shape).T).T
      qstd=X.where(X<=percentile2).std(axis=1,skipna=True,ddof=1)
      qstd=(np.asarray(qstd)*np.ones(X.shape).T).T
      return (X-percentile2)/(qstd+np.finfo(float).eps)
    def fit_transform(self,X,y=None):
      try:
        X=pd.DataFrame(X)
        if X.shape[1] == 1:
          X=X.T
      except:
        pass
      self.fit(X)  
      return self.transform(X)
      
class MeanScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='MeanScaling'
      self.mean=0

    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.mean=X.mean(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X),np.asarray(self.mean)))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 
      
class MedianScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='MedianScaling'
      self.median=0

    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.median=X.median(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X),np.asarray(self.median)))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 
class MaxScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='MaxScaling'
      self.max=0

    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.max=X.max(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X),np.asarray(self.max)))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X)
      
class MeanCentering(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='MeanCentering'
      self.mean=0

    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.mean=X.mean(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.asarray(X)-np.asarray(self.mean))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 

class PoissonScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='PoissonScaling'
      self.mean=0

    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.mean=X.mean(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X),np.sqrt(np.abs(np.asarray(self.mean)))))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 


class ParetoScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='ParetoScaling'
      self.std=0
    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.std=X.std(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X),np.sqrt(np.asarray(self.std))))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X) 

class LevelScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='LevelScaling'
      self.mean=0
    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.mean=X.mean(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X)-np.asarray(self.mean),np.asarray(self.mean)))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X)  
class RangeScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='RangeScaling'
      self.max=0
      self.min=0
      self.mean=0
    def fit(self,X,y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      self.max=X.max(axis=0)
      self.min=X.min(axis=0)
      self.mean=X.mean(axis=0)
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.divide(np.asarray(X)-np.asarray(self.mean),np.asarray(self.max)-np.asarray(self.min)))
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X)

class LogTransform(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='LogTransform'
    def fit(self,X,y=None):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(np.log(np.abs(np.asarray(X)+1))) #absolute to remove negative
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X)

class L2NormScaling(BaseEstimator,TransformerMixin):
    def __init__(self):
      self.__name__='L2NormScaling'
    def fit(self,X,y=None):
      pass
    def transform(self,X, y=None):
      try:
        X=pd.DataFrame(X)
      except:
        pass
      return pd.DataFrame(sklearn.preprocessing.normalize(X,norm='l2')) 
    def fit_transform(self,X,y=None):
      self.fit(X)
      return self.transform(X)


class OPLS(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.W_ortho_ = None
        self.P_ortho_ = None
        self.T_ortho_ = None
        self.x_mean_ = None
        self.y_mean_ = None
        self.x_std_ = None
        self.y_std_ = None
    def fit(self, X, Y):
        try:
            X=np.asarray(X)
        except:
            pass
        try:
            Y=np.asarray(Y)  
        except:
            pass
        if Y.ndim == 1:
            try:
                Y = Y.reshape(-1,1)
            except:
                pass

        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ =X.mean(axis=0)
        self.x_std_ =X.std(axis=0)
        self.y_std_ = X.std(axis=0)

        Z = X.copy()
        w = np.dot(X.T, Y)  
        w /= np.linalg.norm(w)  
        W_ortho = []
        T_ortho = []
        P_ortho = []
        for i in range(self.n_components):
            t = np.dot(Z, w)  
            p = np.dot(Z.T, t) / np.dot(t.T, t).item()  
            w_ortho = p - np.dot(w.T, p).item() / np.dot(w.T, w).item() * w  
            w_ortho = w_ortho / np.linalg.norm(w_ortho)  
            t_ortho = np.dot(Z, w_ortho) 
            p_ortho = np.dot(Z.T, t_ortho) / np.dot(t_ortho.T, t_ortho).item()
            Z -= np.dot(t_ortho, p_ortho.T)
            W_ortho.append(w_ortho)
            T_ortho.append(t_ortho)
            P_ortho.append(p_ortho)
        self.W_ortho_ = np.hstack(W_ortho)
        self.T_ortho_ = np.hstack(T_ortho)
        self.P_ortho_ = np.hstack(P_ortho)
        
    def transform(self, X):
        Z = X
        Z -= self.x_mean_
        for i in range(self.n_components):
            t = np.dot(Z, self.W_ortho_[:, i]).reshape(-1, 1)
            Z -= np.dot(t, self.P_ortho_[:, i].T.reshape(1, -1))
        return Z
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

      
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path=r'C:\Users\User\Downloads\\'
    data=pd.read_excel(path+'Data1.xlsx',index_col=0)
    data=data.iloc[:100,:140]
    
    data2=data.iloc[100:150,:140]
    Y=data.iloc[:100,-1]
    rs=OPLS()
    Yt=rs.fit_transform(data,Y)
    print(Yt)
    
    #print(Y)


    #pd.DataFrame(data1).plot()
    '''
    data.plot(legend=False)
    msc=SNV()
    trans_data=msc.fit_transform(data)
    trans_data=pd.DataFrame(trans_data)
    trans_data.plot(legend=False)
    '''
    plt.show()

