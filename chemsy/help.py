from chemsy.prep.methods import *
from chemsy.predict.methods import *
from inspect import getmembers, isclass


def see_methods(module):
    class_members=getmembers(module, isclass)
    search_list=[]
    for x in class_members:
        search_list.append(x[0])
    delete_list=['C','RBF','ShuffleSplit','TransformerMixin','RegressorMixin','BaselineRemoval','BaseEstimator']    
    output_list= [x for x in search_list if x not in delete_list]
    print('Method supported:')
    print(output_list)
    
    
def visualise_pipeline(ax,
                       custom_recipe: Dict[str, any],
                       solution: SupervisedChemsy,
                       max_solutions: int = np.inf,
                       color_list: List[any] = None):
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
            ax.scatter(clf_n, step_n, c=color_list[index_val], alpha=0.8)

    max_step_n = len(custom_recipe)
    ax.set_xlabel("solution no.")
    ax.set_ylabel("step no.")
    ax.set_yticks(np.linspace(0, max_step_n, max_step_n + 1))

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

    ax_twin.plot(solution.get_results(verbose=False).to_numpy()[:max_solutions, result_col], color='k')
    return ax_twin


def visualise_color(ax,
                    custom_recipe: Dict[str, any],
                    max_chars: int = 10,
                    Capitalize: bool = False,
                    color_list: List[any] = None):
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
                step_str = "skip"
            elif Capitalize:
                step_str = "".join([letter for letter in step_str if letter.isupper()])
            table_vals[level_n + 1][step_n + 1] = step_str[:min(len(step_str), max_chars)]

    table_vals = np.array(table_vals)
    ax.table(cellText=table_vals, cellColours=table_cols, loc='top')
