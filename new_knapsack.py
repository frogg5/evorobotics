import random
import copy
import statistics
import matplotlib.pyplot as plt

global knapsack_weight
global knapsack_values
global max_weight 
global answer

with open('weights.txt', 'r') as file:
    knapsack_weight = [int(line.strip()) for line in file]
with open('values.txt', 'r') as file:
    knapsack_values = [int(line.strip()) for line in file]
with open('max_weight.txt', 'r') as file:
    max_weight = int(file.read().strip())

answer = 309

def generatePopulation(populationsize):
	r = []
	initial_population = []
	for i in range(populationsize):
		r = []
		for j in range(len(knapsack_weight)):
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

def parentselection_tournament(population_values, parameter_parentsamplesize): #tournament style parent selection, selects 2 parents
	k = parameter_parentsamplesize
	candidates = []
	for i in range(k):
		r = random.randint(0, len(population_values) - 1) #population size
		while r in candidates:
			r = random.randint(0, len(population_values) - 1)
		candidates.append(r) #select 5 candidates
	#print("candidates", candidates)
	candidates_values = []
	for candidate in candidates:
		candidates_values.append(population_values[candidate])
	#print("candidates values", candidates_values)
	best_candidate = candidates[candidates_values.index(max(candidates_values))]
	#print("best_candidate_index", best_candidate)
	return best_candidate #returns the index of the best candidate in population_values / population

def mutation_allindex(individual): #selects all indexes, might change each one
	parameter_allindex_random_mutation_rate = 22 #5% mutation rate / gene #the higher this goes up the higher the success rate
	i = 0
	for item in individual:
		#print(i)
		random_prob = random.randint(1, 100)
		#print(random_prob)
		if random_prob <= parameter_allindex_random_mutation_rate:
			#print(individual[i])
			if individual[i] == 0:
				individual[i] = 1
			else:
				individual[i] = 0
		i = i+1
	return individual

def crossover_twopoint(parent1, parent2): #make two individuals not one
	x = random.randint(1,len(parent1)-3)
	y = random.randint(1,len(parent1)-1) #maybe start with x, not 1
	while(y<=x):
		y = random.randint(1,len(parent1)-1)
	#print(x,y)
	newindividual1 = parent1[:x] + parent2[x:y] + parent1[y:]
	newindividual2 = parent2[:x] + parent1[x:y] + parent2[y:]
	return newindividual1, newindividual2

#uniform crossover
#every gene has an equal chance of switching.
#random org - makes a template. 0 1 0 1. But it means a gene from parent 1, parent 2.
#screenshotted, 6/2 for reference

#multipoint - uses a template to determine where crossover happens

#when implementing generational, implement elitism
#elitism 10%. Top 10% from old pop. stay

def graph(listX, listY, nameX, nameY, title):
	plt.plot(listX, listY)
	plt.xlabel(nameX)
	plt.ylabel(nameY)
	plt.title(title)
	plt.show()
	print("graphed")

def main():
	cases_count = 50
	evolutionary_cycle_count = 550

	highest_end_value = []
	percent_count = 0

	for i in range(cases_count):
		paramter_populationsize = 100
		population_individuals = generatePopulation(paramter_populationsize)
		population_values = []
		for individual in population_individuals:
			population_values.append(evaluateIndividual(individual))
		population = list(zip(population_individuals, population_values))
		print("starting population values: ", population_values)
		#print("starting population: ", population)

		for j in range(evolutionary_cycle_count):

			print("CYCLE #", j)
			#Find two parents
			parameter_tournamentsize = 20
			parent1 = parentselection_tournament(population_values, parameter_tournamentsize)
			parent2 = parentselection_tournament(population_values, parameter_tournamentsize)
			while parent1==parent2:
				parent2 = parentselection_tournament(population_values, parameter_tournamentsize)
			print("parents:", parent1, parent2)
			print("parents:", population[parent1], population[parent2])

			#Chance of crossover happening. If crossover happens, then 2 offspring are created and maybe mutated.
			parameter_crossoverchance = 80
			x = random.randint(1,100) #put this inside the crossover function
			if x<=parameter_crossoverchance:
				offspring = crossover_twopoint(population[parent1][0], population[parent2][0])
				offspring1 = copy.deepcopy(offspring[0])
				offspring2 = copy.deepcopy(offspring[1])
				offspring1 = mutation_allindex(offspring1)
				offspring2 = mutation_allindex(offspring2)
				offspring1 = offspring1, evaluateIndividual(offspring1)
				offspring2 = offspring2, evaluateIndividual(offspring2)
				print("offspring 1: ", offspring1)
				print("offspring 2: ", offspring2)

				lowest_individual = min(population_values)
				lowest_individual_index = population_values.index(lowest_individual)
				print("lowest individual index 1: ", lowest_individual_index)
				population_values[lowest_individual_index] = offspring1[1]
				population[lowest_individual_index] = offspring1
				lowest_individual = min(population_values)
				lowest_individual_index = population_values.index(lowest_individual)
				print("lowest individual index 2: ", lowest_individual_index)
				population_values[lowest_individual_index] = offspring2[1]
				population[lowest_individual_index] = offspring2

			#else: the two parents are the offspring without crossover. mutate(?) them. then replaced. restructure ~140-on

			print("end cycle population values:", population_values)
			#print("end cycle population:", population)

		#cycle is over, determine:
		highest_organism_value = max(population_values)
		highest_organism_index = population_values.index(highest_organism_value)
		highest_organism = population[highest_organism_index]
		print("final result population values", population_values)
		print("final result population", population)
		print(highest_organism)
		highest_end_value.append(highest_organism[1])
		if highest_organism[1] == answer:
			percent_count = percent_count + 1

	print("success rate: ", percent_count/cases_count)

	#graphing
	x_axis_trials = []
	for i in range(cases_count):
		x_axis_trials.append(i)
	graph(x_axis_trials, highest_end_value, "Trail #", "End result", "End result vs Trial number")

if __name__ == "__main__":
	main()

#5/27 - look in DEAP
# is my implementation correct
# at what point does it just become random-brute force-until-right?
# how do i implement generational GA
