import matplotlib.pyplot as plt
import csv
import sys

def graph_best_average(filename):
    x = []
    y = []
    y2 = []

    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            x.append(row[0])
            y.append(row[1])
            y2.append(row[2])
    x = list(map(float, x))
    y = list(map(float, y))
    y2 = list(map(float, y2))

    plt.plot(x, y, color = 'g', linestyle = 'dashed',
             marker = 'o',label = "Max organism value")
    plt.plot(x, y2, color = 'r', linestyle = 'dashed',
             marker = 'x',label = "Average organism value")

    plt.xticks(rotation = 25)
    plt.xlabel('Generation')
    plt.ylabel('Max org value')
    plt.title('Max org value vs generation number', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()

def main():
    graph_best_average(sys.argv[1])
    #graph_best_average('data\case3.csv')

if __name__ == "__main__":
    main()
