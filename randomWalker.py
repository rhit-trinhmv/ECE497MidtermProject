# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:43:01 2024

@author: trinhmv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cop
import copy
import random
import numpy as np

class randomWalker():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 1
        
    def step(self):
        direction = random.randint(1, 4)
        if(direction == 1):
            self.y += self.velocity
            if(self.y > 50):
                self.y -= 2*self.velocity
            
        elif(direction == 2):
            self.x += 1
            if(self.x > 50):
                self.x -= 2*self.velocity
        elif(direction == 3):
            self.y -= 1
            if(self.y < -50):
                self.y += 2*self.velocity
        elif(direction == 4):
            self.x -= 1
            if(self.x < -50):
                self.x += 2*self.velocity

# rw = randomWalker()
# xList = []
# yList = []
# xList.append(rw.x)
# yList.append(rw.y)
# for i in range(100):
#     rw.step()
#     xList.append(rw.x)
#     yList.append(rw.y)

# plt.plot(xList, yList)