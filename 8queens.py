import random

def generatePopulation(populationsize):
	initial_population = []
	board = []
	for i in range(populationsize):
		board = []
		for j in range(8):
			r = random.randint(0,7)
			if board!=0:
				while r in board:
					r = random.randint(0, 7)
			else:
				r = random.randint(0, 7)
			board.append(r)
		initial_population.append(board)
	return(initial_population)

initial_population = generatePopulation(20)

#index -> row, list item -> column
def returnAttackingPairs(individual):
	#count diagonals
	diagonal1 = {}
	diagonal2 = {}
	for row, col in enumerate(individual):
		d1 = row - col
		d2 = row + col
		diagonal1[d1] = diagonal1.get(d1,0) + 1
		diagonal2[d2] = diagonal2.get(d2,0) + 1
	def count_pairs(diagonal):
		return sum((count * (count - 1)) for count in diagonal.values() if count > 1)
	#count same column
	samecolumn_count = 0
	for point in individual:
		if point in individual:
			samecolumn_count = samecolumn_count + 1
    #no need for counting same rows
	return count_pairs(diagonal1) + count_pairs(diagonal2) + samecolumn_count

def parentselection(population):
	k = 5
	def generateCandidates(k, exempt, population):
		candidates = []
		for i in range(5):
			r = random.randint(0, len(population)-1) #population size
			while (r in candidates) or (r == exempt):
				r = random.randint(0, 19)
			candidates.append(r) #select 5 candidates
		return candidates
    
print(initial_population[0])
print(returnAttackingPairs(initial_population[0]))
parentselection(initial_population)


#rewrite
#column check, row check, diagonal check
#represent of list of 8 of coordinate points
#fine to generate infeasable solutions
#double counting is fine


#rewrite knapsack into functions
