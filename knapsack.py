import random
import copy

#problem definition
global knapsack_weight
global knapsack_values
knapsack_weight = [1, 1 ,2, 3, 4, 5, 6, 8, 12] #contains 9, index 0-8
knapsack_values = [2, 2 ,4, 7, 3, 6, 5, 10, 15]
global max_weight 
max_weight = 18

#generate population
def generatePopulation(populationsize):
	r = []
	initial_population = []
	for i in range(populationsize):
		r = []
		for j in range(9):
			r.append(random.randint(0,1))
		initial_population.append(r)
	return initial_population

def evaluateIndividual(individual):
	def returnWeight(individual, knapsack_weight):
		weight = 0
		for i in range(9):
			weight = weight + individual[i]*knapsack_weight[i]
		return weight

	def returnValue(individual, knapsack_values):
		value = 0
		for i in range(9):
			value = value + individual[i]*knapsack_values[i]
		return value

	weight = returnWeight(individual, knapsack_weight)
	value = returnValue(individual, knapsack_values)
	if weight > max_weight:
		return 0
	evaluationvalue = value
	return evaluationvalue

#STORE LIST OF 

def parentselection(population): #tournament style parent selection, selects 2 parents
	k = 5
	def generateCandidates(k, exempt, population):
		candidates = []
		for i in range(k):
			r = random.randint(0, len(population) - 1) #population size
			while (r in candidates) or (r in exempt):
				r = random.randint(0, len(population) - 1)
			candidates.append(r) #select 5 candidates
		return candidates
	bestcandidates = []
	for i in range(2): #generate 2 parents
		candidates = generateCandidates(k, bestcandidates, population)
		#print(candidates)
		best_index = candidates[0]
		for candidate in candidates:
			if evaluateIndividual(population[candidate]) > evaluateIndividual(population[best_index]):
				best_index = candidate
		bestcandidates.append(best_index)
	return bestcandidates #returns indexes
	#Roullette wheel selection -> TODO

def parentselection_roulette(population): #roulette style parent selection, selects 1 parent
	# note to self: it may be easier to just select 1 parent at a time, just call the function again until the duplicate parent isnt selected
	fitness_score_sum = 0
	fitness_score_list = []
	for individual in population:
		fitness_score_list.append(evaluateIndividual(individual))
	fitness_score_sum = sum(fitness_score_list)
	pick = random.uniform(0, fitness_score_sum)
	current = 0
	for index, fitnessvalue in enumerate(fitness_score_list):
		current += fitnessvalue
		if current > pick:
			return index
#GENETIC MUTATION
#CROSSOVER

def mutation_oneindex(individual): #selects one index, might change it
	random_index = random.randint(0, len(individual) - 1)
	#probability of mutation is 5%
	random_mutation_rate = 5
	random_prob = random.randint(1, 100)
	#print("random prob", random_prob)
	if random_prob <= random_mutation_rate:
		#print(individual[random_index])
		if individual[random_index] == 0:
			individual[random_index] = 1
		else:
			individual[random_index] = 0
	return individual

def mutation_allindex(individual): #selects all indexes, might change each one
	random_mutation_rate = 5 #.5% mutation rate / gene
	i = 0
	for item in individual:
		#print(i)
		random_prob = random.randint(1, 1000)
		#print(random_prob)
		if random_prob <= random_mutation_rate:
			#print(individual[i])
			if individual[i] == 0:
				individual[i] = 1
			else:
				individual[i] = 0
		i = i+1
	return individual

def replacement(population, individual, evaluation_list):
	#lowest = evaluation_list.index(min(evaluation_list))
	#compare individual with fitness
	ind_fitness_score = evaluateIndividual(individual)
	if ind_fitness_score >= min(evaluation_list):
		lowest = evaluation_list.index(min(evaluation_list))
		population[lowest] = individual
		#print(individual) #TODO: test later
	return population

def main():
	initial_population = generatePopulation(20) #generate population
	max_generation = 100 #termination condition
	i = 0
	evaluation_list = []
	for individual in initial_population:
		print(i, individual, evaluateIndividual(individual))
		evaluation_list.append(evaluateIndividual(individual))
		i = i+1 #print out each individual, index + identity + evaluation
	for j in range(max_generation):
		parent = parentselection(initial_population) #select 2 parents (indexes in a list)
		parent1 = copy.deepcopy(initial_population[parent[0]])
		parent2 = copy.deepcopy(initial_population[parent[1]])
		print(parent1, parent2)
		print(parent)
		print("mutate? parent1", mutation_allindex(parent1), evaluateIndividual(parent1))
		print("mutate? parent2", mutation_allindex(parent2), evaluateIndividual(parent2))
		replacement(initial_population, individual, evaluation_list)
		#TODO return new population
		#TODO crossover, implement 1 point crossover and 2 point crossover. required to pass parent 1 and parent 2. 

if __name__ == "__main__":
	main()
