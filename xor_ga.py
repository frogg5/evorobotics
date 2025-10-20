import numpy as np
import copy
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return z * (1 - z)

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)
Y = np.array([[0],[1],[1],[0]], dtype=int)

graph_x = []
graph_y = []

def generate_population(population_size=100):
    hidden_size = 2
    input_size = 2
    output_size = 1
    population_list = []
    for i in range(population_size): #random weights, random bias generation
        W1 = np.random.randn(input_size, hidden_size) * 0.5
        b1 = np.random.randn(1, hidden_size)
        W2 = np.random.randn(hidden_size, output_size) * 0.5
        b2 = np.random.randn(1, output_size)
        population_list.append([W1, b1, W2, b2])
    return population_list

def evaluateIndividual(W1, b1, W2, b2): #forward pass, returns error
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    error = np.mean(0.5 * (a2 - y)**2)
    return error

def evaluateIndividual_rawweights(W1, b1, W2, b2): #forward pass, returns raw weight
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    error = np.mean(0.5 * (a2 - y)**2)
    return a2

def parentselection_tournament(population_list, number_of_parent_pairs = 1, worst_performing = False): #returns indexes of parent pairs
    error_list = []
    parent_index_pairs = []
    #print(population_list)
    #print(error_list)
    for individual in population_list:
        error_list.append(float(evaluateIndividual(individual[0], individual[1], individual[2], individual[3]))) #W1, b1, W2, b2
    for i in range(int(number_of_parent_pairs)): #normal - select best parents out of pop.
        lowest_two_indexes = [i for i, _ in sorted(enumerate(error_list), key=lambda x: x[1])[:2]]
        parent_index_pairs.append(lowest_two_indexes)
        error_list[lowest_two_indexes[0]], error_list[lowest_two_indexes[1]] = 1, 1
            #print(lowest_indexes)
    return parent_index_pairs

def worstselection(population_list, number_of_pairs = 1): #returns indexes of population members with highest error to be replaced
    error_list = []
    worst_index_pairs = []
    for individual in population_list:
        error_list.append(float(evaluateIndividual(individual[0], individual[1], individual[2], individual[3]))) #W1, b1, W2, b2    
        #print("error list", error_list)
    for i in range(int(number_of_pairs)):
        sorted_indexes = sorted(range(len(error_list)), key=lambda i: error_list[i], reverse=True)
        top_two_indexes = sorted_indexes[:2]
        error_list[top_two_indexes[0]], error_list[top_two_indexes[1]] = 0, 0
        worst_index_pairs.append(top_two_indexes)
    return worst_index_pairs

def crossover_uniform(parent1, parent2): #[w1, b1, w2, b2]
    crossover_chance = 0.8
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    if random.random() < crossover_chance: #80% chance of crossover happening
        for i in range(2):#W1
            for k in range(2):
                #print(child1[0][i][k], "W1")
                if random.random() > 0.5:
                    child1[0][i][k], child2[0][i][k] = child2[0][i][k], child1[0][i][k]
        for i in range(2): #b1
            #print(child1[1][0][i], "b1")
            if random.random() > 0.5:
                child1[1][0][i], child2[1][0][i] = child2[1][0][i], child1[1][0][i]
        for i in range(2): #W2
            #print(child1[2][i][0], "W2")
            if random.random() > 0.5:
                child1[2][i][0], child2[2][i][0] = child2[2][i][0], child1[2][i][0]
        for i in range(1): #b2
            #print(child1[3][i][0], "b2")
            if random.random() > 0.5:
                child1[3][i][0], child2[3][i][0] = child2[3][i][0], child1[3][i][0]
    child1, child2 = mutation_allindex(child1), mutation_allindex(child2)
    return child1, child2

def replacement(population_list, parent_index_pairs): #returns a new population
    new_population = copy.deepcopy(population_list)
    replacable_pair_index = worstselection(population_list, len(parent_index_pairs))
    #print("rep", replacable_pair_index)
    #print("par",parent_index_pairs)
    i = 0
    for parent_pair in parent_index_pairs:
        parent1, parent2 = population_list[parent_pair[0]], population_list[parent_pair[1]]
        child1, child2 = crossover_uniform(parent1, parent2)
        worst_indexes = worstselection(population_list, len(parent_index_pairs))
        #print(worst_indexes, "worst index")
        new_population[replacable_pair_index[i][0]] = child1
        new_population[replacable_pair_index[i][1]] = child2
    return new_population

def mutation_allindex(parent1):
    mutation_chance = 0.05
    child1 = copy.deepcopy(parent1)
    for i in range(2):#W1
        for k in range(2):
                #print(child1[0][i][k], "W1")
            if random.random() < mutation_chance:
                child1[0][i][k] = random.uniform(-1, 1)
    for i in range(2): #b1
            #print(child1[1][0][i], "b1")
        if random.random() < mutation_chance:
            child1[1][0][i] = random.uniform(-1, 1)
    for i in range(2): #W2
            #print(child1[2][i][0], "W2")
        if random.random() < mutation_chance:
            child1[2][i][0] = random.uniform(-1, 1)
    for i in range(1): #b2
            #print(child1[3][i][0], "b2")
        if random.random() < mutation_chance:
                child1[3][i][0] = random.uniform(-1, 1)
    return child1

def main():
    trial_count = 1
    generations = 1000
    population_size = 500
    parent_pool_size = 30 # number of individual parents

    success_count = 0
    for trial in range(trial_count):
        population = generate_population(population_size)
        for generation in range(generations):
            parent_index_list = parentselection_tournament(population, parent_pool_size/2) #number of pairs
            new_population = replacement(population, parent_index_list) #includes crossover + mutation
            population = new_population
            print("gen:",generation, "out of", generations)
        best_fit_individual = population[parentselection_tournament(population, 1)[0][0]]
        rawweight_best_fit_individual = evaluateIndividual_rawweights(best_fit_individual[0], best_fit_individual[1], best_fit_individual[2], best_fit_individual[3])
        print("RAW:", rawweight_best_fit_individual)
        weighted_best_fit_individual = (rawweight_best_fit_individual >= 0.5).astype(int)
        print("Weighted", weighted_best_fit_individual)
        if (np.array_equal(weighted_best_fit_individual, np.array([[0],[1],[1],[0]], dtype=int))):
            success_count+=1
        print(f"trial{trial}")
    print(f"Success rate: {success_count/trial_count}")

if __name__ == "__main__":
    main()
