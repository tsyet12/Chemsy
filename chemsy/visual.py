from chemsy.prep.methods import *
from chemsy.predict.methods import *
from chemsy.explore import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
def visualise_pipeline(ax,
                       custom_recipe,
                       solution,
                       max_solutions= np.inf,
                       color_list = None):
    '''
    a function which creates a scatter plot, which indicates which level correspond to what method;
    the function chemsy.help.visualise_color will show an corresponding table which can be used as a a legend.
    
    :param ax: on which ax the scatter plot should be made
    :param custom_recipe: which custom recipe was used to create the list of pipelines
    :param solution: the SupervisedChemsy instance
    :param max_solutions: the number of solutions which should be plotted
    :param color_list: a list containing the colors for each method number. default are the plt colors
    :return: twin axes of the inputted ax plot.
    '''
    # the colors of the dots
    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for clf_n, clf in enumerate(solution.get_pipelines()):
        if clf_n >= max_solutions:
            break

        for step_n, step in enumerate(clf):
            recipe_vals = custom_recipe[list(custom_recipe.keys())[step_n]]
            recipe_str_vals = [str(val) for val in recipe_vals]
            index_val = recipe_str_vals.index(str(step))
            ax.scatter(clf_n, step_n, c=color_list[index_val], alpha=0.8,marker='s',s=100)

    max_step_n = len(custom_recipe)
    ax.set_xlabel("Solution no.")
    ax.set_ylabel("Step no.")
    ax.set_yticks(np.linspace(0, max_step_n-1, max_step_n ))

    ax_twin = ax.twinx()
    # plot the error / accuracy rate
    if solution.classify:
        # show the cross_val_accuracy
        result_col = 5
        ax_twin.set_ylabel("CV accuracy")
    else:
        # show the MAE
        result_col = 2
        ax_twin.set_ylabel("CV MAE")

    if np.isinf(max_solutions):
        max_solutions = len(solution.get_results())

    ax_twin.plot(np.asarray(solution.get_results(verbose=False))[:max_solutions, result_col], color='k')
    return ax_twin


def visualise_color(ax,
                    custom_recipe,
                    max_chars = 20,
                    Capitalize = False,
                    color_list = None,loc='top'):
    """
    :param ax: on which ax should the plot be made
    :param custom_recipe: which custom recipe was applied to create the list of pipelines
    :param max_chars: how many chars should be displayed per cell
    :param Capitalize: only show capital letters for methods, e.g. PLS(cv=KFold()) becomes PLSKF
    :param color_list: list of colors
    :return:
    """

    # the colors of the dots
    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # the width of the table
    table_width = max([len(val) for val in custom_recipe.values()]) + 1
    table_height = len(custom_recipe) + 1  # leave space for a header

    if table_width - 1 > len(color_list):
        raise ValueError(
            "Not enough colors are available for the amount of unique values per level, "
            "consider inputting a custom color list.")

    table_vals = []
    table_cols = []
    for row in range(table_height):
        cur_row_vals = [f"step {row - 1}"]
        cur_row_col = ['#FFFFFF']
        for col in range(1, table_width):
            cur_row_vals.append("")
            cur_row_col.append('#FFFFFF')
        if row == 0:
            cur_row_vals[0] = 'color:'
            table_cols.append(['#FFFFFF'] + color_list[:table_width - 1])
        else:
            table_cols.append(cur_row_col)
        table_vals.append(cur_row_vals)

    for level_n, level_vals in enumerate(custom_recipe.values()):
        str_level_vals = [str(val) for val in level_vals]
        for step_n, step_str in enumerate(str_level_vals):
            if step_str == "None":
                step_str = "Skip"
            elif Capitalize:
                step_str = "".join([letter for letter in step_str if letter.isupper()])
            table_vals[level_n + 1][step_n + 1] = step_str[:min(len(step_str), max_chars)]

    table_vals = np.array(table_vals)
    ax.table(cellText=table_vals, cellColours=table_cols, loc=loc)

def test_pred_kde(y_test,y_pred,nbins=300,ax=None,verbose=False):
    """
    :param y_test: array of actual output y
    :param y_pred: array of predicted output y by model
    :param nbins: number of bins to control the finess of grid in plot
    :param ax: on which ax should the plot be made
    :return:twin axes of the inputted ax plot.
    """
    x,y=y_test,y_pred
    z=np.concatenate([x,y])
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[z.min():z.max():nbins*1j, z.min():z.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi=zi/np.max(zi)
    f, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel('Actual Data')
    ax.set_ylabel('Model Prediction')
    ax.set_aspect('equal')
    h=ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto',cmap=plt.cm.get_cmap('hot_r'))
    cbar=f.colorbar(h, ax=ax,fraction=0.046, pad=0.04)
    cbar.set_label('Occurance Density', rotation=270, labelpad=20)
    ax.scatter(x,y,color='white',edgecolors='black',label='Test Data')
    ax.plot([0, 1], [0,1],'k--', transform=ax.transAxes)
    ylim=ax.get_ylim()
    xlim=ax.get_xlim()
    ax.set_ylim((z.min(),z.max()))
    ax.set_xlim((z.min(),z.max()))
    f.patch.set_facecolor('white')
    textstr="$n$: "+str(len(x))+'\n$R^2$: '+str(round(r2_score(x,y),3))+'\nRMSE: '+str(round(mean_squared_error(x,y)**0.5,3))+"\nMAE: "+str(round(mean_absolute_error(x,y),3))+"\nMBE: "+str(round(np.sum(x-y)/len(x),3))
    ax.text(0.7,0.05,  textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',bbox=dict(facecolor='white',edgecolor='grey', alpha=0.5,boxstyle='round'))
    if verbose==True:
        plt.rcParams.update({'font.size': 16})
        #plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='smaller')
        plt.rc('ytick', labelsize='smaller')
        plt.rc('text', usetex=False)
        plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.show()
    return ax
    
if __name__ == "__main__":    
    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVR

    X, Y = load_diabetes(return_X_y=True)
    custom_recipe= {
    "Level 0":[None,RobustScaler()],
    "Level 1":[MSC(),StandardScaler(),MinMaxScaler()],
    "Level 2":[PCA()],
    "Level 3":[PartialLeastSquaresCV()]
    }
    solutions=SupervisedChemsy(X, Y,recipe=custom_recipe)
    
    '''
    f, ax = plt.subplots(nrows=2, ncols=1,figsize=(6, 6), gridspec_kw={'height_ratios': [1, 4]})
    visualise_pipeline(ax[1],custom_recipe,solutions)
    visualise_color(ax[0],custom_recipe,loc='center')
    ax[1].invert_yaxis()

    ax[0].axis('off')
    plt.tight_layout()
    plt.show()
    '''
    pipeline=solutions.get_pipelines()[0]
    pipeline.fit(X,Y)
    Y_pred=pipeline.predict(X)
    test_pred_kde(Y,Y_pred.T[0],verbose=True)
    
    


    