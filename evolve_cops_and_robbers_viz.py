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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.patches as patches

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
    
    steps = 30
    
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
    
    steps = 30
    
    c = cop.Cop(cop_start[0], cop_start[1])
    r = robber.robber(money_bag_cords)
    # copXlist = np.zeros(steps + 1)
    # copYlist = np.zeros(steps + 1)
    # robberXlist = np.zeros(steps + 1)
    # robberYlist = np.zeros(steps + 1)
    copXlist = []
    copYlist = []
    robberXlist = []
    robberYlist = []
    copXlist.append(c.x)
    copYlist.append(c.y)
    robberXlist.append(r.x)
    robberYlist.append(r.y)
    
    
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
                    
        copXlist.append(c.x)
        copYlist.append(c.y)
        robberXlist.append(r.x)
        robberYlist.append(r.y)
        
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
    # while copXlist[-1] == 0 and copYlist[-1] == 0:
    #     del(copXlist[-1])
    #     del(copYlist[-1])
    copXlist = np.array(copXlist)
    copYlist = np.array(copYlist)
    robberXlist = np.array(robberXlist)
    robberYlist = np.array(robberYlist)
    
    fig, ax = plt.subplots(figsize=(10,8))
    plt.subplots_adjust(right=0.75) 
    line, = ax.plot([], [], '-o', markersize=2)
    current, = ax.plot([], [], 'go', label='Current Position', markersize=10)
    cline, = ax.plot([], [], '-o', markersize=2)
    ccurrent, = ax.plot([], [], 'bo', label='Current Position', markersize=10)
    
    x_coords, y_coords = zip(*r.money_bag_coords)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.scatter(r.deposit_x, r.deposit_y, color='magenta',s=75, label="Deposit Area")
    ax.scatter(x_coords, y_coords, color='red', s=75, label="Money Bag Locations")
    ax.grid()
    ax.legend(bbox_to_anchor=(1.33, 1), loc='upper right')
    flashlight = patches.Wedge((c.x, c.y), c.distance, 0, 0, color='yellow', alpha=0.3)  # Initial flashlight
    ax.add_patch(flashlight)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cops and Robbers")
    
    def update(frame):
        # Update robber's path
        line.set_data(robberXlist[:frame+1], robberYlist[:frame+1])
        current.set_data(robberXlist[frame], robberYlist[frame])
        
        # Update cop's path
        cline.set_data(copXlist[:frame+1], copYlist[:frame+1])
        ccurrent.set_data(copXlist[frame], copYlist[frame])
        
        flashlight.set_center((copXlist[frame], copYlist[frame]))
        start_angle = np.degrees(c.direction - c.angle)
        end_angle = np.degrees(c.direction + c.angle)
        flashlight.set_theta1(start_angle)
        flashlight.set_theta2(end_angle)
        
        return line, current, cline, ccurrent, flashlight

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(robberXlist), interval=50, blit=True)
    anim.save('cops_and_robbers_animation1.gif', writer=PillowWriter(fps=20))
    
    # for i in range(len(robberXlist)):
    #     line.set_data(robberXlist[:i+2], robberYlist[:i+2])
    #     current.set_data(robberXlist[i], robberYlist[i])
        
    #     cline.set_data(copXlist[:i+2], copYlist[:i+2])
    #     ccurrent.set_data(copXlist[i], copYlist[i])
        
    #     plt.pause(.05)
            
    # plt.plot(copXlist,copYlist)
    # plt.plot(robberXlist,robberYlist)
    # plt.grid()
    # plt.plot(copXlist[0],copYlist[0], 'ro')
    # plt.plot(robberXlist[0],robberYlist[0], 'ko')
    # plt.plot(copXlist[-1],copYlist[-1], 'ro')
    # plt.plot(robberXlist[-1],robberYlist[-1], 'ko')
    # x_coords, y_coords = zip(*r.money_bag_coords)
    # plt.scatter(x_coords,y_coords, color='green', s = 20, label = "Money Bag Locations")
    # plt.scatter(r.deposit_x,r.deposit_y, color = 'gray', s= 25, label = "Deposit Area")
    # plt.legend()
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Cops and Robbers")
    plt.show()

# Visualize data
viz(a)

