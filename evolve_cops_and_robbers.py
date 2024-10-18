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
#dataset = [[-1,-1],[-1,1],[1,-1],[1,1]]
#labels = [0,1,1,0]
# Parameters of the neural network
layers = [21, 40, 40, 2]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.5
mutatProb = 0.01
tournaments = 100*popsize

money_bag_cords = [(10, 10), (20, 30), (-10, -10), (-40, -30),(5, -25), (35, -35), 
            (-25, 10), (-15, 45)]
cop_start = (-50, -50)

def fitnessFunction(genotype):
    # Step 1: Create the neural network.
    a = fnn.FNN(layers)

    # Step 2. Set the parameters of the neural network according to the genotype.
    a.setParams(genotype)
    
    # Step 3. For each training point in the dataset, evaluate the current neural network.
   # error = 0.0
    
    r = robber.robber(money_bag_cords)
    c = cop.Cop(cop_start[0], cop_start[1])
    bags_x = [-1, -1 ,-1 , -1, -1 ,-1]
    bags_y = [ -1, -1, -1, -1, -1, -1]
    deposit_x = -1
    deposit_y = -1
    
    steps = 1000
    
    for i in range(steps):
        for k in range(len(c.moneyBags)):
            bags_x[k] = c.moneyBags[k][0]
            bags_y[k] = c.moneyBags[k][1]
            
        if len(c.depositBox)!=0:
            deposit_x = c.depositBox[0]
            deposit_y = c.depositBox[1]
                    
        
        inputs = [c.x, c.y, c.direction, c.velocity, 
                  c.seen, bags_x[0],bags_y[0],bags_x[1],bags_y[1],bags_x[1],bags_y[1],bags_x[2],bags_y[2],bags_x[3],bags_y[3],
                  bags_x[4],bags_y[4],bags_x[5],bags_y[5], deposit_x, deposit_y]
        
        output = a.forward(inputs)
        
        c.step(output)
        r.step()
        c.look(r.x, r.y, r.money_bag_coords, r.deposit_x, r.deposit_y)
        if c.captured:
            break
        #if(cop.seen)
        # error += np.abs(a.forward(dataset[i]) - labels[i])
    print("fitness:" + str((steps-1-i)/(steps-1)))
    return (steps-1-i)/(steps -1)

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
def viz(neuralnet):
    # X = np.linspace(-1.05, 1.05, 100)
    # Y = np.linspace(-1.05, 1.05, 100)
    # output = np.zeros((100,100))
    # i = 0
    # for x in X: 
    #     j = 0
    #     for y in Y: 
    #         output[i,j] = neuralnet.forward([x,y])
    #         j += 1
    #     i += 1
    # plt.contourf(X,Y,output)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # for i in range(len(dataset)):
    #     if label[i] == 1:
    #         plt.plot(dataset[i][0],dataset[i][1],'wo')
    #     else:
    #         plt.plot(dataset[i][0],dataset[i][1],'wx')
    # plt.show()  
    # weightplots = []
    # for l in range(len(neuralnet.weights)):
    #     for k in range(len(neuralnet.weights[l])):
    #         for z in range(len(neuralnet.weights[l][k])):
    #             weightplots.append(neuralnet.weights[l][k][z])
    
    steps = 1000
    
    c = cop.Cop(cop_start[0], cop_start[1])
    r = robber.robber(money_bag_cords)
    copXlist = np.zeros(steps + 1)
    copYlist = np.zeros(steps + 1)
    robberXlist = np.zeros(steps + 1)
    robberYlist = np.zeros(steps + 1)
    copXlist[0] = c.x
    copYlist[0] = c.y
    robberXlist[0] = r.x
    robberYlist[0] = r.y
    
    
    bags_x = [-1, -1 ,-1 , -1, -1 ,-1]
    bags_y = [ -1, -1, -1, -1, -1, -1]
    deposit_x = -1
    deposit_y = -1
    
    
    for i in range(steps):
        for k in range(len(c.moneyBags)):
            bags_x[k] = c.moneyBags[k][0]
            bags_y[k] = c.moneyBags[k][1]
            
        if len(c.depositBox)!=0:
            deposit_x = c.depositBox[0]
            deposit_y = c.depositBox[1]
                    
        copXlist[i+1] = c.x
        copYlist[i+1] = c.y
        robberXlist[i+1] = r.x
        robberYlist[i+1] = r.y
        
        inputs = [c.x, c.y, c.direction, c.velocity, 
                  c.seen, bags_x[0],bags_y[0],bags_x[1],bags_y[1],bags_x[1],bags_y[1],bags_x[2],bags_y[2],bags_x[3],bags_y[3],
                  bags_x[4],bags_y[4],bags_x[5],bags_y[5], deposit_x, deposit_y]
        
        output = a.forward(inputs)
        
        c.step(output)
        r.step()
        c.look(r.x, r.y, r.money_bag_coords, r.deposit_x, r.deposit_y)
        if c.captured:
            print("CAPTURED!")
            break
    print("steps:" + str(i))
    while copXlist[-1] == 0 and copYlist[-1] == 0:
        del(copXlist[-1])
        del(copYlist[-1])
    plt.plot(copXlist,copYlist)
    plt.plot(robberXlist,robberYlist)
    plt.grid()
    plt.plot(copXlist[0],copYlist[0], 'ro')
    plt.plot(robberXlist[0],robberYlist[0], 'ko')
    plt.plot(copXlist[-1],copYlist[-1], 'ro')
    plt.plot(robberXlist[-1],robberYlist[-1], 'ko')
    x_coords, y_coords = zip(*r.money_bag_coords)
    plt.scatter(x_coords,y_coords, color='green', s = 20, label = "Money Bag Locations")
    plt.scatter(r.deposit_x,r.deposit_y, color = 'gray', s= 25, label = "Deposit Area")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cops and Robbers")
    plt.show()

# Visualize data
viz(a)

