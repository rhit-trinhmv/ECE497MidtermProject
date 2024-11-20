# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:23:04 2024

@author: ahmadst1
"""

import numpy as np

class BasicCop:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        #self.angle  = np.pi/8
        #self.distance = 15
        self.direction = 2
        self.velocity = 3  
       #self.seen = 0
        self.captured = False
        #self.moneyBags = []
        #self.depositBox = []
        #self.max_y = self.y + np.arctan(self.angle)*self.distance
        #self.min_y = self.y - np.arctan(self.angle)*self.distance
        #self.max_x = self.x + self.distance
        #self.min_x = self.x
        
    def step(self,output):
        maximum = output[0][0]
        self.direction = 1
        if(output[0][1]>=maximum):
            maximum = output[0][1]
            self.direction = 2
        if(output[0][2]>=maximum):
            maximum = output[0][2]
            self.direction = 3
        if(output[0][3]>=maximum):
            maximum = output[0][3]
            self.direction = 4
        
        # self.direction = np.round(3*output[0][0] + 1,0)
        # if self.seen:
        #     self.velocity = np.round(np.min(4*[output[0][1]+1,5]),0)
        # else:
        #     self.velocity = np.round(np.min(2*[output[0][1]+1,3]),0)
            
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
    

    def look(self, r_x, r_y):
        if np.abs(r_x-self.x)<=1 and np.abs(r_y-self.y)<=1:
            self.captured = True
       
                                               