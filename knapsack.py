import random

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

initial_population = generatePopulation(20)

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

def evaluateIndividual(individual):
	weight = returnWeight(individual, knapsack_weight)
	value = returnValue(individual, knapsack_values)
	if weight > max_weight:
		return 0
	evaluationvalue = weight + value
	return evaluationvalue

def parentselection(population): #tournament style parent selection, selects 2 parents
	k = 5
	def generateCandidates(k, exempt, population):
		candidates = []
		for i in range(5):
			r = random.randint(0, len(population) - 1) #population size
			while (r in candidates) or (r == exempt):
				r = random.randint(0, 19)
			candidates.append(r) #select 5 candidates
		return candidates
	candidates = generateCandidates(k, 999, population)
	bestcandidates = []
	for i in range(2):
		selected_candidate = 0
		for candidate in candidates:
			if evaluateIndividual(population[candidate]) > selected_candidate:
				selected_candidate = candidate
		bestcandidates.append(selected_candidate)
		candidates = generateCandidates(k, selected_candidate, population)
	return bestcandidates

#chosenpopulation = random.randint()
print(parentselection(initial_population))
#print(initial_population)
#print(initial_population_knapsack_weight)
#print(initial_population_knapsack_values)