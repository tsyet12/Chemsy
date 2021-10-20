from evoopt.solvers.GeneticAlgorithm import GeneticAlgorithm
		
if __name__=="__main__":
    def f(x1,x2):
        return (x1+5)*(x2+5)**2

    GA=GeneticAlgorithm(f,['x1','x2'],[-100,-100],[100,100])
    GA.solve()
    GA.plot_result()
    
	