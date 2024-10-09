# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:41:34 2024

@author: trinhmv
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class robber():
    
    def __init__(self, money_bag_coords):
        self.x = 3
        self.y = 3
        self.velocity = 2
        self.deposit_size = 4
        self.deposit_coords = [(x, y) for x in 
                           range(-int(self.deposit_size / 2), int(self.deposit_size / 2) + 1) for y in
                           range(-int(self.deposit_size / 2), int(self.deposit_size / 2) + 1)]
        # self.money_bag_coords = [(20, 20), (40, 60), (-20, -20), (-80, -60),
        #                          (10, -50), (70, -70), (-50, 20), (-30, 90)]
        self.money_bag_coords = money_bag_coords
        self.direction = 2 # 1 up, 2 right, 3 down, 4 left
        self.seen = 0
        self.money_bags = 0
        self.deposited = 0
        self.captured = False
        
        
    def step(self, direction, cop):
        self.direction = direction
        if(cop.seen == 1):
            self.seen == 1
        if(self.direction == 1):
            self.y += self.velocity
        elif(self.direction == 2):
            self.x += self.velocity
        elif(self.direction == 3):
            self.y -= self.velocity
        elif(self.direction == 4):
            self.x -= self.velocity
            
        if((self.x + 1, self.y + 1) in self.deposit_coords or
           (self.x - 1, self.y + 1) in self.deposit_coords or
           (self.x - 1, self.y - 1) in self.deposit_coords or
           (self.x + 1, self.y - 1) in self.deposit_coords):
            self.deposited += self.money_bags
            self.money_bags = 0
            
        if((self.x, self.y) in self.money_bag_coords):
            self.money_bags += 1
            self.money_bag_coords.remove((self.x, self.y))

g = robber([(20, 20), (40, 60), (-20, -20), (-80, -60),(10, -50), (70, -70), 
            (-50, 20), (-30, 90)])

fig, ax = plt.subplots()
x_coords, y_coords = zip(*g.money_bag_coords)
square = patches.Rectangle(g.deposit_coords[0], g.deposit_size, g.deposit_size, linewidth=2, edgecolor='blue', facecolor='blue')
ax.add_patch(square)
plt.scatter(x_coords, y_coords, color='red', s=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()