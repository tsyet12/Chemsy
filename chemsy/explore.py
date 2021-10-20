from sklearn import svm
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
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
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import savgol_filter
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from functools import reduce
from chemsy.prep.methods import SavgolFilter, BaselineASLS, BaselineModpoly, BaselineZhangFit, BaselineIModPoly, BaselineLinear, BaselineSecondOrder, MSC, FirstDerivative, SNV
from chemsy.predict.methods import PartialLeastSquares
import pprint
def MBEfunc(y_true , y_pred):
    '''
    Calculates mean bias error
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Bias score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)   
    diff = (y_true-y_pred)
    mbe = diff.mean()
    return mbe

def fix_name(score_dict):
    '''
    Takes the score dict and renames the keys properly
    Parameters:
        score_dict (dict): default score dictionary
    Returns:
        renamed_dict (dict): renamed score dictionary
    '''
    d=score_dict
    d1 = {'test_MAE':'cross_val_MAE', 'test_MSE':'cross_val_MSE', 'test_R2':'cross_val_R2','test_MBE':'cross_val_MBE','fit_time':'fit_time','score_time':'score_time'}
    renamed_dict = dict((d1[key], value) for (key, value) in d.items())
    return renamed_dict

def get_name(method_list):
    '''
    Takes list that contains the instances of multiple method objects (for modelling pipeline), and extract their names, then combine them in a string.
    Parameters:
        method_list (list): List of method objects
    Returns:
        pipeline_name (string): The combined name of the pipeline (e.g. "Method1 + Method 2 + Method 3 + ..." )
    '''
    nlist=[]
    for x in method_list:
      nlist.append(str(type(x).__name__))
    while 'NoneType' in nlist: nlist.remove('NoneType')
    nstr=''
    kk=0
    while kk in range(len(nlist)):
        nstr=nstr+nlist[kk]+" + "
        kk+=1
    pipeline_name=nstr[:-3]    
    return pipeline_name


def get_name_sep(method_list):
    '''
    Takes list that contains the instances of multiple method objects (for modelling pipeline), and extract their names, then return them in a string list.
    Parameters:
        method_list (list): List of method objects
    Returns:
        pipeline_name (list): The combined name of the pipeline (e.g. [Method1, Method 2, Method 3, ...] )
    '''
    nlist=[]
    for x in method_list:
      nlist.append(str(type(x).__name__))
    nlist = ["None" if x=="NoneType" else x for x in nlist]
    kk=0
    pipeline_name=[]
    while kk in range(len(nlist)):
        pipeline_name.append(nlist[kk])
        kk+=1 
    return pipeline_name


class SupervisedChemsy():
    def __init__(self,X,y,cv=None,random_state=999,verbose=False, path='./', recipe='normal',solver=None,output_csv=False):
        self.df=None
        self.pbar=None
        self.recipe=recipe
        self.ExploreModel(X,y,cv=None,random_state=999,verbose=False, path='./', recipe=recipe)
        if not verbose:
            warnings.filterwarnings("ignore")
        self.pbar.refresh()
    def get_results(self):
        print(self.df)
        return self.df
        
    def get_recipe(self):
        d={k: get_name_sep(v) for k, v in self.recipe.items()}
        print("\n")
        for k, v in d.items():
            print(k, v)
        return d
    def ExploreModel(self,X,y,cv=None,random_state=999,verbose=False, path='./', recipe='normal',solver=None,output_csv=False):
      if cv==None:
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
      kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

      if recipe=='normal':
        prep=[StandardScaler(),MinMaxScaler(),RobustScaler(), MaxAbsScaler()]
        trans=[PowerTransformer(),QuantileTransformer(output_distribution='normal', random_state=0), PCA(n_components='mle'), KernelPCA(kernel='polynomial') , None]
        model=[svm.SVR(), 
          DecisionTreeRegressor() ,
          GaussianProcessRegressor(kernel=kernel,alpha=1, n_restarts_optimizer=10), 
          PartialLeastSquares(cv=cv), 
          GradientBoostingRegressor(random_state=random_state),
          RandomForestRegressor(random_state=random_state),
          LinearRegression(),
          KernelRidge(alpha=1.0),
          MLPRegressor(solver='adam', activation='relu'),
          KNeighborsRegressor(),
          Ridge(),
          ElasticNet(),
          Lasso(),
          BayesianRidge()
          ]
      elif recipe=='spectral':
        prep=[BaselineASLS(),BaselineModpoly(),BaselineIModPoly(),BaselineZhangFit(),BaselineLinear(), BaselineSecondOrder(),FirstDerivative()]
        trans=[SavgolFilter(),QuantileTransformer(output_distribution='normal', random_state=0),KernelPCA(kernel='polynomial'),SNV(),MSC()]
        model=[KernelRidge(alpha=1.0),DecisionTreeRegressor(),PartialLeastSquares(cv=cv),ElasticNet(),Lasso(),KNeighborsRegressor()]
      elif recipe=='fast_spectral':
        prep=[BaselineLinear(), BaselineSecondOrder(),FirstDerivative()]
        trans=[SavgolFilter(), KernelPCA(kernel='polynomial'),SNV(),MSC()]
        model=[KernelRidge(alpha=1.0),PartialLeastSquares(cv=cv),ElasticNet(),Lasso(),KNeighborsRegressor()]
      elif recipe=='medium_spectral':
        prep=[BaselineLinear(), BaselineSecondOrder(),FirstDerivative()]
        trans=[SavgolFilter(),KernelPCA(kernel='polynomial'),SNV(),MSC()]
        model=[svm.SVR(), 
          DecisionTreeRegressor() ,
          GaussianProcessRegressor(kernel=kernel,alpha=1, n_restarts_optimizer=10), 
          PartialLeastSquares(cv=cv), 
          GradientBoostingRegressor(random_state=random_state),
          RandomForestRegressor(random_state=random_state),
          LinearRegression(),
          KernelRidge(alpha=1.0),
          MLPRegressor(solver='adam', activation='relu'),
          KNeighborsRegressor(),
          Ridge(),
          ElasticNet(),
          Lasso(),
          BayesianRidge()
          ]
      else:
        pass

      MBE=make_scorer(MBEfunc, greater_is_better=False)
      scoring_dict={'MAE':'neg_mean_absolute_error','MSE':'neg_mean_squared_error','R2':'r2','MBE':MBE}

      recipe_list=list(recipe.values())
      block_names=list(recipe.keys())
      structure_list=[]
      for element in recipe_list:
        structure_list.append(len(element)-1)
      n_space=np.prod(np.asarray(structure_list)+1)
      print('\033[1m' +'Total Model Search Space= ' + str(n_space))
      self.pbar = tqdm(total=n_space,leave=True,position=0)
      
      
      def model_optimize(eval_list):
        feed_list=[0]*len(recipe_list)
        self.pbar.update(1)
        for ii in range(0,len(recipe_list)):
          feed_list[ii]=recipe_list[ii][int(eval_list[ii])]
        method_name=get_name(feed_list)  
        try:
          clf=make_pipeline(*feed_list)
          score=cross_validate(clf, X, Y, cv=cv,scoring=scoring_dict)
          score=fix_name(score)
          if self.df is None:

            self.df=pd.DataFrame.from_dict(score).mean(axis=0).to_frame().transpose().rename(index={0:method_name})
          else:
            self.df=pd.concat([self.df,pd.DataFrame.from_dict(score).mean(axis=0).to_frame().transpose().rename(index={0:method_name})])
        except Exception as e:
            if verbose:
              print("Fail: "+method_name)
              print(e)
        return score['cross_val_MAE'].mean()

      def default_solver(model,x,xmin,xmax,*args,**kwargs):
        x_best=None
        y_best=99999999999
        x_guess=xmin
        converge=False
        flag=False
        while not converge:
          y=model(x_guess)
          if y<y_best:
            y_best=y
            x_best=x
          #full design
          if flag==True:
            converge=True
          g=0
          while g<len(xmin):
            if x_guess[g]<xmax[g]:
              x_guess[g]=x_guess[g]+1
              x_guess[:g]=np.zeros_like(x_guess[:g])
              g=999999999 #escape
            else:
              g=g+1 #iterate
          flag=np.array_equal(x_guess,xmax)      
        return x_best, y_best

      
      if solver==None:
        solver=default_solver
      up_bound=np.asarray(structure_list)
      x_best=solver(model_optimize,block_names,xmin=np.zeros_like(up_bound),xmax=up_bound)

      
      self.df.index.name = 'Methods'
      self.df.replace(-np.inf, -999999999999, inplace=True)
      self.df.replace(np.inf,  +999999999999, inplace=True)
      self.df["cross_val_MAE"]=-self.df["cross_val_MAE"]
      self.df["cross_val_MSE"]=-self.df["cross_val_MSE"]
      self.df.sort_values(by=["cross_val_MAE"])
      self.df=self.df.sort_values(by=["cross_val_R2"],ascending=False)
      if output_csv:
        self.df.to_csv(path+'results.csv')
      return self.df
      
if __name__ == "__main__":    
    from sklearn.datasets import load_diabetes
    X, Y = load_diabetes(return_X_y=True)
    custom_recipe= {
    "Level 1":[BaselineASLS(),StandardScaler(),MinMaxScaler(),RobustScaler()],
    "Level 2":[PowerTransformer(),QuantileTransformer(output_distribution='normal', random_state=0), PCA(n_components='mle')],
    "Level 3":[PartialLeastSquares()]
    }
    solutions=SupervisedChemsy(X, Y,recipe=custom_recipe)
    solutions.get_results()