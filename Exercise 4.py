from deap import base
from deap import tools
import matplotlib.pyplot as plt
import random
from deap import creator


COUNTING_ONES = 100
POP_SIZE = 200
CROSS_PROB = 0.9
MUTATION_PROB = 1.0 / COUNTING_ONES
GENERATIONS = 1500
toolbox = base.Toolbox()
toolbox.register("randomOneOrZero",random.randint, 0, 1)
creator.create("MaxFitness",base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individual",tools.initRepeat, creator.Individual, toolbox.randomOneOrZero, COUNTING_ONES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def oneMaxFitness(individual):
    return sum(individual), 

toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / COUNTING_ONES)
toolbox.register("crossover", tools.cxOnePoint)

 
population = toolbox.population(n=POP_SIZE)
interations_number = 0

fitnessValues = list(map(toolbox.evaluate, population))
for individual, fitnessValue in zip(population, fitnessValues) :
    individual.fitness.values = fitnessValue

fitnessValues = [individual.fitness.values[0] for individual in population]

maxFitnessValues = []
meanFitnessValues = []

while max(fitnessValues) < COUNTING_ONES and interations_number < GENERATIONS:
    interations_number = interations_number + 1
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSS_PROB:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
        
    for mutant in offspring:
        if random.random() < MUTATION_PROB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
        
    newIndividuals = [ind for ind in offspring if not ind.fitness.valid]
    newFitnessValues = list(map(toolbox.evaluate, newIndividuals))
    for individual, fitnessValue in zip(newIndividuals, newFitnessValues):
        individual.fitness.values = fitnessValue
    population[:] = offspring
    
    fitnessValues = [ind.fitness.values[0] for ind in population]
    
    maxFitness = max(fitnessValues)
    meanFitness = sum (fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print ("-Generation {} : Maximum Fitness = {}".format(interations_number, maxFitness))
    
    
    best_index = fitnessValues.index(max(fitnessValues))
    print("Optimal  = ", *population[best_index], "\n")
    
    plt.plot(maxFitnessValues, color='blue')
    plt.xlabel('Iterations number')
    plt.ylabel('Highest Fitness')
    plt.title('MAX Best fitness over Generations')
    plt.show()
