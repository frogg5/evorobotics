import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # weights bias, same as xor 2
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        # forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):

        #m = X.shape[0] #not sure what this does entirely
        # output layer error
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        # hiden layer error
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X)):
                # Forward pass for single sample
                x_sample = X[i:i+1]#keep shape
                y_sample = y[i:i+1]
                output = self.forward(x_sample)
                
                #loss
                loss = np.mean((output - y_sample) ** 2)
                total_loss += loss
                # Backward pass and update weights
                self.backward(x_sample, y_sample, learning_rate)
            
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            #if epoch % 1000 == 0:
                #print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        return losses

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
# true table
y = np.array([[0],
              [1],
              [1],
              [0]])



# Create and train network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X, y, epochs=1000, learning_rate=0.1)  #epochs was actually 1000
#9/22 - try running this 100 times (trials)
#9/22 - also get success rate
#9/22 - accuracy in generation 1-1000. when does convergence tend to occur?

print("\nFinal predictions:")
for i in range(len(X)):
    prediction = nn.forward(X[i:i+1])
    print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {prediction[0][0]:.6f}")



import matplotlib.pyplot as plt
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch') #9/22 - epochs on x axis
plt.ylabel('Loss') #9/22 - freq on y axis
plt.show()

#next step - replace back propogation with genetic algorithms for optimal weights