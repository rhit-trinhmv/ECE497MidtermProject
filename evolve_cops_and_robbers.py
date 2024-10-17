##################################################################################
# Example script for evolving a feedforward neural network to solve XOR problem 
#
# Eduardo Izquierdo
# September 2024
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import fnn 
import ea
import robber
import cop

# Parameters of the XOR task
dataset = [[-1,-1],[-1,1],[1,-1],[1,1]]
labels = [0,1,1,0]
# Parameters of the neural network
layers = [7, 10, 10, 4]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.5
mutatProb = 0.01
tournaments = 100*popsize

money_bag_cords = [(20, 20), (40, 60), (-20, -20), (-80, -60),(10, -50), (70, -70), 
            (-50, 20), (-30, 90)]
cop_start = (10, -10)

def fitnessFunction(genotype):
    # Step 1: Create the neural network.
    a = fnn.FNN(layers)

    # Step 2. Set the parameters of the neural network according to the genotype.
    a.setParams(genotype)
    
    # Step 3. For each training point in the dataset, evaluate the current neural network.
    error = 0.0
    
    robber = robber(money_bag_cords)
    cop = Cop(cop_start[0], cop_start[1])
    
    steps = 100
    
    for i in range(steps):
        
        inputs = [robber.x, robber.y, cop.x, cop.y, 
                  robber.money_bag_cords[0][0], robber.money_bag_cords[0][1],
                  robber.deposit_x, robber.deposit_y]
        
        output = a.forward(inputs)
        direction = np.argmax(output) + 1
        
        robber.step(direction, cop)
        cop.look(robber.x, robber.y, robber.money_bag_cords, robber.deposit_x, robber.deposit_y)
        if(cop.seen)
        # error += np.abs(a.forward(dataset[i]) - labels[i])

    return 1 - (error/len(dataset))

# Evolve
ga = ea.MGA(fitnessFunction, genesize, popsize, recombProb, mutatProb, tournaments)
ga.run()
ga.showFitness()

# Obtain best agent
best = int(ga.bestind[-1])
print(best)
a = fnn.FNN(layers)
a.setParams(ga.pop[best])

# Function to visualize the best solution
def viz(neuralnet, dataset, label):
    X = np.linspace(-1.05, 1.05, 100)
    Y = np.linspace(-1.05, 1.05, 100)
    output = np.zeros((100,100))
    i = 0
    for x in X: 
        j = 0
        for y in Y: 
            output[i,j] = neuralnet.forward([x,y])
            j += 1
        i += 1
    plt.contourf(X,Y,output)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(len(dataset)):
        if label[i] == 1:
            plt.plot(dataset[i][0],dataset[i][1],'wo')
        else:
            plt.plot(dataset[i][0],dataset[i][1],'wx')
    plt.show()  
    weightplots = []
    for l in range(len(neuralnet.weights)):
        for k in range(len(neuralnet.weights[l])):
            for z in range(len(neuralnet.weights[l][k])):
                weightplots.append(neuralnet.weights[l][k][z])
    plt.plot(weightplots)
    plt.xlabel("Neural Connection")
    plt.ylabel("Weight")
    plt.title("Evolved Weights of Neural Connections")
    plt.show()

# Visualize data
viz(a, dataset, labels)

