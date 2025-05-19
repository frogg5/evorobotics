import random
import copy
import statistics

#problem definition
global knapsack_weight
global knapsack_values
global max_weight 

with open('weights.txt', 'r') as file:
    knapsack_weight = [int(line.strip()) for line in file]
with open('values.txt', 'r') as file:
    knapsack_values = [int(line.strip()) for line in file]
with open('max_weight.txt', 'r') as file:
    max_weight = int(file.read().strip())

    #OPTIMAL 5/18 - 309

#knapsack_weight = [1, 1 ,2, 3, 4, 5, 6, 8, 12] #contains 9, index 0-8
#knapsack_values = [2, 2 ,4, 7, 3, 6, 5, 10, 15]
#max_weight = 18

#knapsack_weight = [23, 31 ,29, 44, 53, 38, 63, 85, 89, 82] #contains 9, index 0-8
#knapsack_values = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
#max_weight = 165

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
		for i in range(len(individual)):
			weight = weight + individual[i]*knapsack_weight[i]
		return weight

	def returnValue(individual, knapsack_values):
		value = 0
		for i in range(len(individual)):
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
	random_mutation_rate = 5 #5% mutation rate / gene
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

#5/18/25 ??? working properly?
def replacement(population, offspring):
	#lowest = evaluation_list.index(min(evaluation_list))
	#compare individual with fitness
	ind_fitness_score = evaluateIndividual(offspring)
	#print("ind fit score", offspring, ind_fitness_score)
	lowest_key = min(population, key = lambda k:population[k][1])
	lowest = population[lowest_key]
	if ind_fitness_score <= lowest[1]:
		population[lowest_key] = (copy.deepcopy(offspring), ind_fitness_score)
	return population

def crossover_onepoint(parent1, parent2): #make two individuals one
	x = random.randint(1,len(parent1)-1)
	newindividual = parent1[:x] + parent2[x:]
	return newindividual

def crossover_twopoint(parent1, parent2): #make two individuals not one
	x = random.randint(1,len(parent1)-3)
	y = random.randint(1,len(parent1)-1)
	while(y<=x):
		y = random.randint(1,len(parent1)-1)
	#print(x,y)
	newindividual1 = parent1[:x] + parent2[x:y] + parent1[y:]
	newindividual2 = parent2[:x] + parent1[x:y] + parent2[y:]
	return newindividual1, newindividual2

def main():

	def generateEvalutationList(population):
		evaluation_list = []
		i = 0
		for individual in population:
			#print(i, individual, evaluateIndividual(individual))
			evaluation_list.append(evaluateIndividual(individual))
			i = i+1 #print out each individual, index + identity + evaluation
		return evaluation_list

	best_organism_value = []
	for k in range(500): #5/18 was 20, not
		initial_population = generatePopulation(100) #generate population
		max_generation = 100 #termination condition

		evaluation_list = []
		
		evaluation_list = generateEvalutationList(initial_population)
		pop = {i:(j,k) for i,(j,k) in enumerate(zip(initial_population,evaluation_list))}

		population = initial_population

		for j in range(10):
			parent = parentselection(initial_population) #select 2 parents (indexes in a list)
			parent1 = copy.deepcopy(initial_population[parent[0]])
			parent2 = copy.deepcopy(initial_population[parent[1]])
			parent1 = mutation_allindex(parent1)
			parent2 = mutation_allindex(parent2)

			#TODO return new population
			offspring = crossover_twopoint(parent1, parent2)
			#print("offspring:", offspring)
			offspring1 = copy.deepcopy(offspring[0])
			offspring2 = copy.deepcopy(offspring[1])
			offspring1 = mutation_allindex(offspring1)
			offspring2 = mutation_allindex(offspring2)
			offspring = [offspring1, offspring2]

			if population == replacement(pop, offspring[0]): #???? I think something is going wrong here
				print("population the same")
			else:
				print("population is different")
			population = replacement(pop, offspring[0]) #5/4 - not working - need help with
			population = replacement(pop, offspring[1])

			#print(population)
		best_key = max(population, key = lambda k:population[k][1])
		best_organism = population[best_key]	
		print(best_organism, k)
		best_organism_value.append(best_organism[1])
	#print(best_organism_value)
	print("Average best org value is: ", sum(best_organism_value) / len(best_organism_value))	


		#for future reference -> maybe use dictionary where the key is the index, value is a tuple which is the individual, and the next part of the tuple is the fitness
		#or class invidiual -> properties, fitness value. population is made out of the

	#parent1 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
	#parent2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	#print(crossover_twopoint(parent1, parent2))

if __name__ == "__main__":
	main()


#ROS install 5/12
#5/12 try to optimize, test it out. parameter fine tuning, look at parameters, try to find optimal ones
#https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html
#read a file instead of list for knapsack problem
