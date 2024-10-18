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
layers = [19, 40, 40,40,40,40,40,40,40,40,40, 2]
layers_robber = [21, 40, 40,40,40,40,40,40,40,40,40, 2]

# Parameters of the evolutionary algorithm
genesize_cop = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
genesize_robber = np.sum(np.multiply(layers_robber[1:],layers_robber[:-1])) + np.sum(layers_robber[1:]) 
# print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.5
mutatProb = 0.01
tournaments = 100*popsize

money_bag_cords = [(10, 10), (20, 30), (-10, -10), (-40, -30),(5, -25), (35, -35)]
cop_start = (-50, -50)

def fitnessFunction(genotype_c, genotype_r):
    # Step 1: Create the neural network.
    a = fnn.FNN(layers)
    b = fnn.FNN(layers_robber)

    # Step 2. Set the parameters of the neural network according to the genotype.
    a.setParams(genotype_c)
    b.setParams(genotype_r)
    
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
                  c.seen, bags_x[0],bags_y[0],bags_x[1],bags_y[1],bags_x[2],bags_y[2],bags_x[3],bags_y[3],
                  bags_x[4],bags_y[4],bags_x[5],bags_y[5], deposit_x, deposit_y]
        inputs_robber = [r.x,r.y,r.direction,r.velocity,r.seen, money_bag_cords[0][0],money_bag_cords[0][1],
                         money_bag_cords[1][0],money_bag_cords[1][1],money_bag_cords[2][0],money_bag_cords[2][1],
                         money_bag_cords[3][0],money_bag_cords[3][1],money_bag_cords[4][0],money_bag_cords[4][1],
                         money_bag_cords[5][0],money_bag_cords[5][1],r.deposit_x,r.deposit_y, r.money_bags, r.deposited]
        
        
        output = a.forward(inputs)
        output_robber = b.forward(inputs_robber)
        
        c.step(output)
        r.step(output_robber)
        c.look(r.x, r.y, r.money_bag_coords, r.deposit_x, r.deposit_y)
        r.look(c.seen)
        if c.captured:
            break
        if(r.deposited == 6):
            break
        #if(cop.seen)
        # error += np.abs(a.forward(dataset[i]) - labels[i])
    # print("fitness:" + str((steps-1-i)/(steps-1)))
    if c.captured:
        c_fitness = (steps-1-i)/(steps-1)
        r_fitness = 1/6 * r.deposited
    elif r.deposited == 6:
        c_fitness = 0
        r_fitness = 1
    else:
        c_fitness = 0
        r_fitness = 1/6 * r.deposited
        
    return [c_fitness, r_fitness]

# Evolve
ga = ea.MGA(fitnessFunction, genesize_cop,genesize_robber, popsize, recombProb, mutatProb, tournaments)
ga.run()
ga.showFitness()

# Obtain best agent
best_robber = int(ga.bestind_r[-1])
best_cop = int(ga.bestind_c[-1])
# print(best)
a = fnn.FNN(layers)
b = fnn.FNN(layers_robber)
a.setParams(ga.pop_cop[best_cop])
b.setParams(ga.pop_robber[best_robber])



# Function to visualize the best solution
def viz():

    steps = 1000
    
    c = cop.Cop(cop_start[0], cop_start[1])
    r = robber.robber(money_bag_cords)
    # copXlist = np.zeros(steps + 1)
    # copYlist = np.zeros(steps + 1)
    # robberXlist = np.zeros(steps + 1)
    # robberYlist = np.zeros(steps + 1)
    # copXlist[0] = c.x
    # copYlist[0] = c.y
    # robberXlist[0] = r.x
    # robberYlist[0] = r.y
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
         
        
        
        
        inputs = [c.x, c.y, c.direction, c.velocity, 
                 c.seen, bags_x[0],bags_y[0],bags_x[1],bags_y[1],bags_x[2],bags_y[2],bags_x[3],bags_y[3],
                 bags_x[4],bags_y[4],bags_x[5],bags_y[5], deposit_x, deposit_y]
        inputs_robber = [r.x,r.y,r.direction,r.velocity, r.seen, money_bag_cords[0][0],money_bag_cords[0][1],
                        money_bag_cords[1][0],money_bag_cords[1][1],money_bag_cords[2][0],money_bag_cords[2][1],
                        money_bag_cords[3][0],money_bag_cords[3][1],money_bag_cords[4][0],money_bag_cords[4][1],
                        money_bag_cords[5][0],money_bag_cords[5][1],r.deposit_x,r.deposit_y, r.money_bags, r.deposited]
        
        output = a.forward(inputs)
        output_robber = b.forward(inputs_robber)
        
        c.step(output)
        r.step(output_robber)
        
        # copXlist[i+1] = c.x
        # copYlist[i+1] = c.y
        # robberXlist[i+1] = r.x
        # robberYlist[i+1] = r.y
        copXlist.append(c.x)
        copYlist.append(c.y)
        robberXlist.append(r.x)
        robberYlist.append(r.y)
        
        c.look(r.x, r.y, r.money_bag_coords, r.deposit_x, r.deposit_y)
        r.look(c.seen)
        if c.captured:
            print("CAPTURED!")
            break
        if(r.deposited == 6):
            print("MONEY STOLEN")
            break

    print("steps:" + str(i))
    # while copXlist[-1] == 0 and copYlist[-1] == 0:
    #     copXlist = copXlist[:-1]
    #     copYlist = copYlist[:-1]
    # while robberXlist[-1] == 0 and robberYlist[-1] == 0:
    #     robberXlist = robberXlist[:-1]
    #     robberYlist = robberYlist[:-1]
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
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
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
        current.set_data((robberXlist[frame],), (robberYlist[frame],))
        
        # Update cop's path
        cline.set_data(copXlist[:frame+1], copYlist[:frame+1])
        ccurrent.set_data((copXlist[frame],), (copYlist[frame],))
        
        flashlight.set_center((copXlist[frame], copYlist[frame]))
        start_angle= -0.393
        end_angle = 0.393
        # if c.direction == 1:
        #     start_angle = 1.178
        #     end_angle = 1.963
        # elif c.direction == 2:
        #     start_angle = 5.89
        #     end_angle = 6.675
        # elif c.direction == 3:
        #     start_angle = 4.32
        #     end_angle = 5.105
        # else:
        #     start_angle = 157.5
        #     end_angle = 202.5
        flashlight.set_theta1(start_angle)
        flashlight.set_theta2(end_angle)
        
        return line, current, cline, ccurrent, flashlight

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(robberXlist), interval=50, blit=True)
    anim.save('cops_and_robbers_animation_GAY_12.gif', writer=PillowWriter(fps=20))
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
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Cops and Robbers")
    plt.show()

# Visualize data
viz()

