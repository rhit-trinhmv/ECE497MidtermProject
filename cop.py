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
        self.max_y = self.y + np.arctan(self.angle)*self.distance
        self.min_y = self.y - np.arctan(self.angle)*self.distance
        self.max_x = self.x + self.distance
        self.min_x = self.x
        
    def step(self,output):
        self.direction = np.round(3*output[0][0] + 1,0)
        if self.seen:
            self.velocity = np.round(np.min(4*[output[0][1]+1,5]),0)
        else:
            self.velocity = np.round(np.min(2*[output[0][1]+1,3]),0)
            
        # print("Output Velocity: " + str(output[0][1]))
        # print("Actual Velocity: "+str(self.velocity))
        
        # print("Output Direction: " + str(output[0][0]))
        # print("Actual Direction: "+str(self.direction))
        
        if self.direction==2:
            self.x += self.velocity
            if self.x>50:
                self.x -= 2*self.velocity
                self.direction = 4
        elif self.direction == 3:
            self.y -= self.velocity
            if self.y<-50:
                self.y += 2*self.velocity
                self.direction = 1
        elif self.direction==1:
            self.y += self.velocity
            if self.y>50:
                self.y -= 2*self.velocity
                self.direction = 3
        else:
            self.x -= self.velocity
            if self.x<-50:
                self.x += 2*self.velocity
                self.direction = 2
    
    def look(self, r_x, r_y, bags, d_x, d_y):
        if self.direction == 2:
            self.max_y = self.y + np.arctan(self.angle)*self.distance
            self.min_y = self.y - np.arctan(self.angle)*self.distance
            self.max_x = self.x + self.distance
            self.min_x = self.x
        elif self.direction ==3:
            self.max_x = self.x + np.arctan(self.angle)*self.distance
            self.min_x = self.x - np.arctan(self.angle)*self.distance
            self.max_y = self.y 
            self.min_y = self.y - self.distance
        elif self.direction ==1:
            self.max_x = self.x + np.arctan(self.angle)*self.distance
            self.min_x = self.x - np.arctan(self.angle)*self.distance
            self.max_y = self.y + self.distance
            self.min_y = self.y
        else:
            self.max_y = self.y + np.arctan(self.angle)*self.distance
            self.min_y = self.y - np.arctan(self.angle)*self.distance
            self.max_x = self.x 
            self.min_x = self.x - self.distance
        if r_x > self.min_x and r_x < self.max_x and r_y > self.min_y and r_y < self.max_y:
            self.seen = 1
        else:
            self.seen = 0
        if np.abs(r_x-self.x)<=1 and np.abs(r_y-self.y)<=1:
            self.captured = True
        for bag in bags:
            if bag[0] > self.min_x and bag[0] < self.max_x and bag[1] > self.min_y and bag[1] <= self.max_y:
                if [bag[0],bag[1]] not in self.moneyBags:
                    self.moneyBags.append([bag[0],bag[1]])
        if d_x > self.min_x and d_x < self.max_x and r_y > self.min_y and r_y < self.max_y:
            self.depositBox.append(d_x)
            self.depositBox.append(d_y)
                                           
            
            
        