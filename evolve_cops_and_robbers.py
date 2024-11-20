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
import eaOriginal
import robber
import cop
import BasicCop
import randomWalker
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.patches as patches



# Parameters of the neural network
layers = [4,2, 2, 2, 2,  4]
robber_locations = [[25, 25]]
# robber_locations = [[25, 25],[-25,-25]]
# robber_locations = [[25, 25],[-25,-25],[-25,25],[25,-25]]
# robber_locations = [[25, 25],[-25,-25],[-25,25],[25,-25],[0, 10],[0,-10],[-10,0],[10,0]]
# robber_locations = [[25, 25],[-25,-25],[-25,25],[25,-25],[0, 10],[0,-10],[-10,0],[10,0],[-40,20],[-40,-20],[40,20],[40,-20]]
# robber_locations = [[25, 25],[-25,-25],[-25,25],[25,-25],[0, 10],[0,-10],[-10,0],[10,0],[-40,20],[-40,-20],[40,20],[40,-20], [20,-40],[20,40],[-20,40],[-20,-40]]
#layers_robber = [21, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 2]

# Parameters of the evolutionary algorithm
genesize_cop = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
#genesize_robber = np.sum(np.multiply(layers_robber[1:],layers_robber[:-1])) + np.sum(layers_robber[1:]) 
# print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.5
mutatProb = 0.01
tournaments = 100*popsize

money_bag_cords = [(10, 10), (20, 30), (-10, -10), (-40, -30),(5, -25), (35, -35)]
cop_start = (0, 0)

def fitnessFunction(genotype_c):
    # Step 1: Create the neural network.
    a = fnn.FNN(layers)
#    b = fnn.FNN(layers_robber)

    # Step 2. Set the parameters of the neural network according to the genotype.
    a.setParams(genotype_c)
    # b.setParams(genotype_r)
    
    # Step 3. For each training point in the dataset, evaluate the current neural network.
   # error = 0.0
    
#    r = robber.robber(money_bag_cords)
#    c = cop.Cop(cop_start[0], cop_start[1])
    
    
#    bags_x = [-1, -1 ,-1 , -1, -1 ,-1]
#    bags_y = [ -1, -1, -1, -1, -1, -1]
#    deposit_x = -1
#    deposit_y = -1
    c_fitness = 0
    steps = 1000
    for k in range(len(robber_locations)):
        c = BasicCop.BasicCop(cop_start[0], cop_start[1])
        r = randomWalker.randomWalker(robber_locations[k][0], robber_locations[k][1])
        distance = 0
        for i in range(steps):
            # for k in range(len(c.moneyBags)):
            #     bags_x[k] = c.moneyBags[k][0]
            #     bags_y[k] = c.moneyBags[k][1]
                
            # if len(c.depositBox)!=0:
            #     deposit_x = c.depositBox[0]
            #     deposit_y = c.depositBox[1]
                        
            
            #inputs = [c.x, c.y, c.direction, c.velocity, 
                      # c.seen, bags_x[0],bags_y[0],bags_x[1],bags_y[1],bags_x[2],bags_y[2],bags_x[3],bags_y[3],
                      # bags_x[4],bags_y[4],bags_x[5],bags_y[5], deposit_x, deposit_y]
            #inputs_robber = [r.x,r.y,r.direction,r.velocity,r.seen, money_bag_cords[0][0],money_bag_cords[0][1],
                             # money_bag_cords[1][0],money_bag_cords[1][1],money_bag_cords[2][0],money_bag_cords[2][1],
                             # money_bag_cords[3][0],money_bag_cords[3][1],money_bag_cords[4][0],money_bag_cords[4][1],
                             # money_bag_cords[5][0],money_bag_cords[5][1],r.deposit_x,r.deposit_y, r.money_bags, r.deposited]
            inputs = [c.x, c.y, r.x, r.y]
            
            output = a.forward(inputs)
            # output_robber = b.forward(inputs_robber)
            
            c.step(output)
            # r.step(output_robber)
            c.look(r.x, r.y)
            # r.look(c.seen)
            distance = np.abs(r.x-c.x) + np.abs(r.y-c.y)
            if c.captured:
                break
            # if(r.deposited == 6):
            #     break
            #if(cop.seen)
            # error += np.abs(a.forward(dataset[i]) - labels[i])
        # print("fitness:" + str((steps-1-i)/(steps-1)))
        # if c.captured:
        #     c_fitness = (steps-1-i)/(steps-1)
        #     r_fitness = 1/6 * r.deposited
        # elif r.deposited == 6:
        #     c_fitness = 0
        #     r_fitness = 1
        # else:
        #     c_fitness = 0
        #     r_fitness = 1/6 * r.deposited
               
        # c_fitness += 1/(distance/(i+1))
        if c.captured:
            c_fitness += 1 + 1/(i+1)
        else:
            c_fitness += 1/(distance) + 1/(i+1)
        
        
    return c_fitness/len(robber_locations)

# Evolve
ga = eaOriginal.MGA(fitnessFunction, genesize_cop, popsize, recombProb, mutatProb, tournaments)
ga.run()
ga.showFitness()

# Obtain best agent
# best_robber = int(ga.bestind_r[-1])
best_cop = int(ga.bestind[-1])
# print(best)
a = fnn.FNN(layers)
# b = fnn.FNN(layers_robber)
a.setParams(ga.pop[best_cop])
# b.setParams(ga.pop_robber[best_robber])



def testpoints():
    total = 0
    radius = [10, 25, 40]
    for l in range(len(radius)):
        rob_x = 0
        rob_y = radius[l]
        for i in range(radius[l]*4):
            steps = 1000
            c = BasicCop.BasicCop(cop_start[0], cop_start[1])
            r = randomWalker.randomWalker(rob_x,rob_y)
            if(rob_x<radius[l] and rob_x>=0 and rob_y>0 and rob_y<=radius[l]):
                rob_x+=1
                rob_y-=1
            elif(rob_y>-radius[l] and rob_y<=0 and rob_x>0 and rob_x<=radius[l]):
                rob_x-=1
                rob_y-=1
            elif(rob_x>-radius[l] and rob_x<=0 and rob_y<0 and rob_y>=-radius[l]):
                rob_x-=1
                rob_y+=1
            else:
                rob_x+=1
                rob_y+=1
            for k in range(steps):
                inputs = [c.x, c.y, r.x, r.y]
                output = a.forward(inputs)
              
                
                c.step(output)
                c.look(r.x, r.y)
               
                if c.captured:
                    print("CAPTURED!")
                    total+=1
                    break
    print("total: " + str(total) + "/300")
        
        

# Function to visualize the best solution
def viz():

    steps = 1000
    
    c = BasicCop.BasicCop(cop_start[0], cop_start[1])
    r = randomWalker.randomWalker(13,25)
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
    # flashlist = []
    # if c.direction == 1:
        # flashlist.append([67.5, 112.5])
        
    # elif c.direction == 2:
        # flashlist.append([-22.5, 22.5])
        
    # elif c.direction == 3:
        # flashlist.append([-67.5, -112.5])
        
    # else:
        # flashlist.append([157.5, 202.5])
        
    
    # bags_x = [-1, -1 ,-1 , -1, -1 ,-1]
    # bags_y = [ -1, -1, -1, -1, -1, -1]
    # deposit_x = -1
    # deposit_y = -1
    
    
    for i in range(steps):
        # for k in range(len(c.moneyBags)):
            # bags_x[k] = c.moneyBags[k][0]
            # bags_y[k] = c.moneyBags[k][1]
            
        # if len(c.depositBox)!=0:
            # deposit_x = c.depositBox[0]
            # deposit_y = c.depositBox[1]
         
        
        
        
        # inputs = [c.x, c.y, c.direction, c.velocity, 
                 # c.seen, bags_x[0],bags_y[0],bags_x[1],bags_y[1],bags_x[2],bags_y[2],bags_x[3],bags_y[3],
                 # bags_x[4],bags_y[4],bags_x[5],bags_y[5], deposit_x, deposit_y]
        # inputs_robber = [r.x,r.y,r.direction,r.velocity, r.seen, money_bag_cords[0][0],money_bag_cords[0][1],
                        # money_bag_cords[1][0],money_bag_cords[1][1],money_bag_cords[2][0],money_bag_cords[2][1],
                        # money_bag_cords[3][0],money_bag_cords[3][1],money_bag_cords[4][0],money_bag_cords[4][1],
                        # money_bag_cords[5][0],money_bag_cords[5][1],r.deposit_x,r.deposit_y, r.money_bags, r.deposited]
        
        inputs = [c.x, c.y, r.x, r.y]
        output = a.forward(inputs)
        # output_robber = b.forward(inputs_robber)
        
        c.step(output)
        # r.step(output_robber)
        
        # copXlist[i+1] = c.x
        # copYlist[i+1] = c.y
        # robberXlist[i+1] = r.x
        # robberYlist[i+1] = r.y
        copXlist.append(c.x)
        copYlist.append(c.y)
        robberXlist.append(r.x)
        robberYlist.append(r.y)
        # if c.direction == 1:
            # flashlist.append([67.5, 112.5])
            
        # elif c.direction == 2:
            # flashlist.append([-22.5, 22.5])
            
        # elif c.direction == 3:
            # flashlist.append([-112.5, -67.5])
            
        # else:
            # flashlist.append([157.5, 202.5])
        
        c.look(r.x, r.y)
        # r.look(c.seen)
        if c.captured:
            print("CAPTURED!")
            break
        # if(r.deposited == 6):
            # print("MONEY STOLEN")
            # break

    print("steps:" + str(i))
    print("CopX:" + str(copXlist[-1]))
    print("CopY:" + str(copYlist[-1]))
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
    # flashlist = np.array(flashlist)
    fig, ax = plt.subplots(figsize=(10,8))
    plt.subplots_adjust(right=0.75) 
    line, = ax.plot([], [], '-o', markersize=2)
    current, = ax.plot([], [], 'go', label='Robber Current Position', markersize=10)
    cline, = ax.plot([], [], '-o', markersize=2)
    ccurrent, = ax.plot([], [], 'bo', label='Cop Current Position', markersize=10)
    
    # x_coords, y_coords = zip(*r.money_bag_coords)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    # ax.scatter(r.deposit_x, r.deposit_y, color='magenta',s=75, label="Deposit Area")
    # ax.scatter(x_coords, y_coords, color='red', s=75, label="Money Bag Locations")
    ax.grid()
    ax.legend(bbox_to_anchor=(1.33, 1), loc='upper right')
    # flashlight = patches.Wedge((c.x, c.y), c.distance, 0, 0, color='yellow', alpha=0.3)  # Initial flashlight
    # ax.add_patch(flashlight)
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
        
        # flashlight.set_center((copXlist[frame], copYlist[frame]))
        # start_angle= -0.393
        # end_angle = 0.393
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
        # flashlight.set_theta1(flashlist[frame][0])
        # flashlight.set_theta2(flashlist[frame][1])
        
        return line, current, cline, ccurrent

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(robberXlist), interval=50, blit=True)
    anim.save('cops_and_robbers_animation_Final_oneTrainingSuccess_test.gif', writer=PillowWriter(fps=20))
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
    # plt.show()

# Visualize data
viz()

#  Test around 100 points
# testpoints()

