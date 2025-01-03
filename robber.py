# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:41:34 2024

@author: trinhmv
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cop
import copy
import random
import numpy as np

class robber():
    
    def __init__(self, money_bag_coords):
        self.x = 3
        self.y = 3
        self.velocity = 2
        self.deposit_x = 0
        self.deposit_y = 0
        # self.deposit_size = 4
        # self.deposit_coords = [(x, y) for x in 
        #                    range(-int(self.deposit_size / 2), int(self.deposit_size / 2) + 1) for y in
        #                    range(-int(self.deposit_size / 2), int(self.deposit_size / 2) + 1)]
        # self.money_bag_coords = [(20, 20), (40, 60), (-20, -20), (-80, -60),
        #                          (10, -50), (70, -70), (-50, 20), (-30, 90)]
        self.money_bag_coords = copy.deepcopy(money_bag_coords)
        self.direction = 2 # 1 up, 2 right, 3 down, 4 left
        self.seen = 0
        self.money_bags = 0
        self.deposited = 0
        self.captured = False
        
        
    def step(self, output):
        self.direction = np.round(3*output[0][0] + 1,0)
        
        self.velocity = np.round(np.min([output[0][1]+1,2]),0)
        # if((self.x + 1, self.y + 1) in self.deposit_coords or
        #    (self.x - 1, self.y + 1) in self.deposit_coords or
        #    (self.x - 1, self.y - 1) in self.deposit_coords or
        #    (self.x + 1, self.y - 1) in self.deposit_coords):
        #     self.deposited += self.money_bags
        #     self.money_bags = 0
    # def step(self):
    #     self.direction = random.randint(1, 4)
        
        if(self.direction == 1):
            self.y += self.velocity
            if(self.y > 50):
                self.y -= 2*self.velocity
                self.direction = 3
        elif(self.direction == 2):
            self.x += self.velocity
            if(self.x > 50):
                self.x -= 2*self.velocity
                self.direction = 4
        elif(self.direction == 3):
            self.y -= self.velocity
            if(self.y < -50):
                self.y += 2*self.velocity
                self.direction = 1
        elif(self.direction == 4):
            self.x -= self.velocity
            if(self.x < -50):
                self.x += 2*self.velocity
                self.direction = 2
        
    def look(self, seen):
        if(self.x == self.deposit_x and self.y == self.deposit_y):
            self.deposited += self.money_bags
            self.money_bags = 0
            
        if((self.x, self.y) in self.money_bag_coords):
            self.money_bags += 1
            self.money_bag_coords.remove((self.x, self.y))  
        
        if seen:
            self.seen = 1
        else: 
            self.seen = 0
        
       
# g = robber([(20, 20), (40, 60), (-20, -20), (-30, -30),(10, -50), (10, -10), 
#             (-50, 20), (-30, 50)])
# xlist = np.zeros(10001)
# ylist = np.zeros(10001)

# fig, ax = plt.subplots()
# x_coords, y_coords = zip(*g.money_bag_coords)
# square = patches.Rectangle(g.deposit_coords[0], g.deposit_size, g.deposit_size, linewidth=2, edgecolor='blue', facecolor='blue')
# ax.add_patch(square)
# plt.figure(figsize=(8, 8))
# plt.scatter(g.deposit_x, g.deposit_y, s=10, label="Deposit Area")
# plt.scatter(x_coords, y_coords, color='red', s=10, label="Money Bag Locations")
# plt.title("Initial Location of Money Bags and Deposit Area")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# fig, ax = plt.subplots()
# line, = ax.plot([], [], '-o')
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.scatter(g.deposit_x, g.deposit_y, color='green',s=40, label="Deposit Area")
# ax.scatter(x_coords, y_coords, color='red', s=40, label="Money Bag Locations")
# for i in range(10000):
#     g.step()
#     xlist[i+1] = g.x
#     ylist[i+1] = g.y
    
#     line.set_data(xlist[:i+2], ylist[:i+2])
#     plt.pause(.01)


# plt.show() 