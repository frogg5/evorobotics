#(reworked from class node to using arrays and numpy)
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return z * (1 - z)

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)
Y = np.array([[0],[1],[1],[0]], dtype=int)

graph_x = []
graph_y = []

hidden_size = 2
input_size = 2
output_size = 1
learning_rate = 3 #was 0.01
epochs = 2000 #was 9999

numberoftrials = 10

for i in range(numberoftrials):
    #define weights and bias with numpy
    W1 = np.random.randn(input_size, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.5
    b2 = np.zeros((1, output_size))

    #maybe dont use batch training
    for e in range(epochs):
        #forward pass
        z1 = X.dot(W1) + b1 #pass into hidden layer
        a1 = sigmoid(z1) #then activate
        z2 = a1.dot(W2) + b2 #pass into output layer
        a2 = sigmoid(z2) #then activate
        #calculate error
        error = np.mean(0.5 * (a2 - y)**2)


        # backpropagation - compares how each weight contributed to error via derivatives
        dL_da2 = (a2 - y) / X.shape[0] #gradient
        dz2 = dL_da2 * sigmoid_derivative(a2) #applies chain rule
        dW2 = a1.T.dot(dz2) 
        db2=np.sum(dz2, axis=0, keepdims=True)
        dz1=dz2.dot(W2.T) * sigmoid_derivative(a1)
        dW1=X.T.dot(dz1)
        db1=np.sum(dz1, axis=0, keepdims=True)

        #update parameters
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        

        #print("Raw outputs at end (a2):")
        #print(a2.round(5))
        raw_weights = a2.round(5)
        print(raw_weights)
        end_weights = (a2 >= 0.5).astype(int)
        print(end_weights)

        if (np.array_equal(end_weights, Y)):
            graph_x.append(i)
            graph_y.append(e)
            print(graph_x)
            print(graph_y)
            break

average = sum(graph_y) / len(graph_y)
median = np.median(np.array(graph_y))
success_rate = len(graph_y) / numberoftrials

print(f"average {average}, median {median}, success_rate {success_rate}")
#avg 694, median 499

'''
import matplotlib.pyplot as plt

plt.plot(graph_x, graph_y, marker='o', label="y = 2x")  # line plot with markers
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("title")
plt.legend()
plt.grid(True)

plt.show()

#try one by one maybe, not batch
#come up with graphs - limit to 1000 maybe - how soon do i get the result

#import matplotlib.pyplot as plt
'''