import numpy as np
import matplotlib.pyplot as plt
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor
import graphviz

def exp_func(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)
exp = make_function(function=exp_func, name='expo', arity=1)

def _fitness(y, y_pred, sample_weight):
    return np.sum(np.abs(y-y_pred))
fit = make_fitness(function=_fitness, greater_is_better=False, wrap=False)

def get_data():
    x = np.linspace(-1, 1, 21).reshape(-1,1)
    y = np.array([0, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784, -0.2289, -0.1664, -0.0909, 0.0, 0.1111, 0.2496, 0.4251, 0.6496, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4.0000] )
    return x, y

pop_size = 1000
function_set = ['add', 'sub', 'mul', 'log', exp, 'sin', 'cos', 'div']
num_generations = 50
crossover_prob = 0.7
mutation_prob = 0.1

def experiment(seed, i):
    est_gp = SymbolicRegressor(population_size = pop_size,
                               generations=num_generations, stopping_criteria=0.01,
                               p_crossover=crossover_prob, p_subtree_mutation=mutation_prob,
                               p_hoist_mutation=mutation_prob, p_point_mutation=mutation_prob,
                               function_set = function_set,
                               max_samples=0.9, verbose=1,
                               metric=fit, random_state=seed)

    est_gp.fit(x, y)
    

    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.xlabel('Generations', fontsize=24)
    plt.ylabel('Best fitness', fontsize=24)
    plt.plot(est_gp.run_details_['best_fitness'], linewidth=3.0)
    plt.grid()

    plt.subplot(1,2,2)
    plt.xlabel('Generations', fontsize=24)
    plt.ylabel('Best size', fontsize=24)
    plt.plot(est_gp.run_details_['best_length'], linewidth=3.0, color='red')
    plt.grid()

    plt.suptitle('Run {}'.format(i), fontsize=24)
    plt.savefig('plot_{}.eps'.format(seed))
    return est_gp.run_details_

if __name__ == '__main__':
    seeds = list(range(1335, 1345))
    x, y = get_data()
    run_details = []
    for i, seed in enumerate(seeds):
        print('Experiment {}: Seed {}'.format(i, seed))
        experiment(seed, i)
        print('\n\n')
