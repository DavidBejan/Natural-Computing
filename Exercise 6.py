import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1337)
POP_SIZE = 10

def plot_route(coord, route, title="Map of cities", is_save=False, filename=""):
    plt.scatter(coord[:, 0], coord[:, 1], s=5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    
    prev_coords = coord[route[0]-1]
    colors = np.linspace(0, 1, len(coord), True)
    for i in range(len(coord)):
        coords = coord[route[i]-1]
        x1, y1 = prev_coords
        x2, y2 = coords
        plt.plot([x1, x2], [y1, y2], 'k-', c=(colors[i], colors[len(coord)-1-i], 0))
        prev_coords = coords
    
    if is_save:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def plot_statistics(fitness_vals, title="", is_save=False, filename=""):
    plt.plot(fitness_vals[0], label="memetic")
    plt.plot(fitness_vals[1], label="normal")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness (total Euclidean distance)")
    plt.legend()
    plt.title(title)
    plt.grid(True)

    if is_save:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()

def eucl_dist(coord, city1, city2):
    x1, y1 = coord[city1-1]
    x2, y2 = coord[city2-1]
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def total_eucl_dist(route, coord):
    return sum([eucl_dist(coord, route[i-1], route[i]) for i in range(1,len(route))])

def swap(route, idx1, idx2):
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def init_population(population_size, n_cities):
    return np.array([np.random.permutation(np.arange(1,n_cities+1)) for _ in range(population_size)])

def fitness(population, coord):
    return np.array([total_eucl_dist(route, coord) for route in population])

def select_parents(population, fitness):
    parents = []
    for _ in range(len(population)):
        # Binary tournament selection
        c1, c2 = np.random.randint(len(population), size=2)
        fittest = population[c1] if fitness[c1] < fitness[c2] else population[c2]
        parents.append(fittest)
    return np.array(parents)

def crossover(parents):
    n_cities = len(parents[0])
    children = []

    for _ in range(len(parents)):
        parents_crossover = parents[np.random.randint(len(parents), size=2)]
        # Randomly determine which parent is cur_parent and which parent is cur_parent
        if np.random.random() < 1:
            parents_crossover = np.flipud(parents_crossover)
            
        cur_parent = parents_crossover[0]
        other_parent = parents_crossover[1]
        child = np.zeros(n_cities, dtype=int)
        
        cut_points = np.random.randint(0,n_cities+1,2)
        if cut_points[0] > cut_points[1]:
            cut_points = np.flip(cut_points)
        cut_p1, cut_p2 = cut_points
        child[cut_p1:cut_p2] = cur_parent[cut_p1:cut_p2]
        # Take set difference such that there are no duplicate values, set assume_uniqe=True to prevent sorting
        remaining_cities = np.setdiff1d(other_parent, child, assume_unique=True)
        child[np.where(child==0)] = remaining_cities
        children.append(child)
    
    return np.array(children)

def mutation(children, mutation_probability=0.5):
    for child in children:
        if np.random.random() < mutation_probability:
            p1, p2 = np.random.choice(len(child), size=2, replace=False)
            child[p1], child[p2] = child[p2], child[p1]
    return children


def local_search(initial_route, coord):
    best_fitness = total_eucl_dist(initial_route, coord)
    best_route = initial_route
    nbs = neighbours(initial_route)
    fitness_nbs = fitness(nbs, coord)
    best_nb_idx = np.argmax(fitness_nbs)

    while fitness_nbs[best_nb_idx] < best_fitness:
        best_fitness = fitness_nbs[best_nb_idx]
        best_tour = nbs[best_nb_idx]
        
        nbs = neighbours(best_route)
        fitness_nbs = fitness(nbs, coord)
        best_nb_idx = np.argmax(fitness_nbs)
    
    return best_route


def neighbours(route):
    neighbours = []
    for i in range(len(route)):
        for j in range(i,len(route)):
            neighbour = swap(route, i, j)
            neighbours.append(neighbour)
    return neighbours

def evolve(coord, is_memetic=False, iterations=10000, step_size=100, population_size=10, mutation_probability=0.5):
    start_time = time.time()
    avg_fitnesses, best_fitnesses = [], []
    population = init_population(population_size, len(coord))
    if is_memetic:
        population = [local_search(route, coord) for route in population]
    
    for i in range(iterations):
        fitness_vals = fitness(population, coord)
        avg_fitnesses.append(np.mean(fitness_vals))
        best_fitnesses.append(np.min(fitness_vals))
        parents = select_parents(population, fitness_vals)
        children = crossover(parents)
        population = mutation(children, mutation_probability)
        if is_memetic:
            population = [local_search(route, coord) for route in population]

        if i % step_size == 0:
            print(i)

    return population, avg_fitnesses, best_fitnesses, time.time()-start_time

def main():
    dataset = "ulysses22"
    iterations = 1000
    pop_size = 10
    mut_prob = 0.05

    coordinates_file = open("data/"+dataset+".tsp.txt", "r")
    coord = np.array([line.split(" ")[1:] for line in coordinates_file.readlines()], dtype=int)
    opt_route = np.array(open("data/"+dataset+".opt.tour.txt", "r").readlines(), dtype=int)

    final_pop_meme, avg_fit_meme, best_fit_meme, time_meme = evolve(coord, is_memetic=True, iterations=iterations, population_size=pop_size, mutation_probability=mut_prob)
    final_fit_meme = fitness(final_pop_meme, coord)
    fittest_meme = final_pop_meme[np.argmin(final_fit_meme)]

    final_pop_norm, avg_fit_norm, best_fit_norm, time_norm = evolve(coord, iterations=iterations, population_size=pop_size, mutation_probability=mut_prob)
    final_fit_norm = fitness(final_pop_norm, coord)
    fittest_norm = final_pop_norm[np.argmin(final_fit_norm)]

    plot_route(coord, fittest_meme, title="Best fitness of final memetic population",
               is_save=True, filename="img/"+dataset+"_tour_meme.png")
    plot_route(coord, fittest_norm, title="Best fitness of final simple population",
               is_save=True, filename="img/"+dataset+"_tour_norm.png")
    plot_route(coord, opt_route, title="Optimal tour for "+dataset, is_save=True, filename="img/"+dataset+"_opt_tour.png")

    plot_statistics([avg_fit_meme, avg_fit_norm],
                    "Average fitness between memetic and simple", is_save=True, filename="img/"+dataset+"_avg_fit_vals.png")
    plot_statistics([best_fit_meme, best_fit_norm],
                    "Best fitness between memetic simple algorithm", is_save=True, filename="img/"+dataset+"_best_fit_vals.png")

if __name__ == "__main__":
    main()
