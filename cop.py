# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:37:04 2024

@author: ahmadst1
"""

import numpy as np


class Cop:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle  = np.pi/8
        self.distance = 15
        self.direction = 2
        self.velocity = 3  
        self.seen = 0
        self.captured = False
        self.moneyBags = []
        self.depositBox = []
        
    def step(self):
        if self.direction==2:
            self.x += self.velocity
        if self.direction==4:
            self.x -= self.velocity
        if self.direction==1:
            self.y += self.velocity
        if self.direction==2:
            self.y -= self.velocity
    
    def look(self, r_x, r_y, bags, d_x, d_y):
        max_y = self.y + np.arctan(self.angle)*self.distance
        min_y = self.y - np.arctan(self.angle)*self.distance
        max_x = self.x + self.distance
        min_x = self.x
        if r_x > min_x and r_x < max_x and r_y > min_y and r_y <max_y:
            self.seen = 1
        if np.abs(r_x-self.x)<=1 and np.abs(r_y-self.y)<=1:
            self.captured = True
        for bag in bags:
            if bag[0] > min_x and bag[0] < max_x and bag[1] > min_y and bag[1] <= max_y:
                if [bag[0],bag[1]] not in self.moneyBags:
                    self.moneyBags.append([bag[0],bag[1]])
        if d_x > min_x and d_x < max_x and r_y > min_y and r_y <max_y:
            self.depositBox.append(d_x)
            self.depositBox.append(d_y)
                                           
            
            
        