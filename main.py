from sklearn.datasets import load_diabetes
import chemsy
from chemsy.explore import SupervisedChemsy
from chemsy.prep.methods import *
from chemsy.predict.methods import *
from chemsy.help import see_methods
if __name__=="__main__":
    from sklearn.datasets import load_diabetes
    X, Y = load_diabetes(return_X_y=True)
    custom_recipe= {
    "Level 1":[BaselineASLS(),StandardScaler(),MinMaxScaler(),RobustScaler()],
    "Level 2":[PowerTransformer(),QuantileTransformer(output_distribution='normal', random_state=0), PCA(n_components='mle')],
    "Level 3":[PartialLeastSquaresCV()]
    }
    solutions=SupervisedChemsy(X, Y,recipe=custom_recipe)
    solutions.get_results()
    
    
    BL=BaselineASLS()
    X_=BL.fit_transform(X=X,y=Y)
    see_methods(chemsy.predict.methods)
