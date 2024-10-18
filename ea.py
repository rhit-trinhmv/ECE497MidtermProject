import numpy as np
import matplotlib.pyplot as plt

class MGA():

    def __init__(self, fitnessfunction, genesize_c, genesize_r, popsize, recomprob, mutationprob, tournaments):
        self.genesize_c = genesize_c
        self.genesize_r = genesize_r
        self.popsize = popsize
        self.recomprob = recomprob
        self.mutationprob = mutationprob
        self.tournaments = tournaments
        self.fitnessfunction = fitnessfunction
        self.pop_cop = np.random.random((popsize,genesize_c))*2 - 1
        self.pop_robber = np.random.random((popsize,genesize_r))*2 - 1
        fits = self.calculateFitness()
        self.fit_cop = fits[0]
        self.fit_robber = fits[1]
        # stats
        gens = tournaments//popsize      
        self.bestfit_c = np.zeros(gens)
        self.avgfit_c = np.zeros(gens)
        self.worstfit_c = np.zeros(gens)
        self.bestind_c = np.zeros(gens)
        self.variance_c = np.zeros(gens)
        
        self.bestfit_r = np.zeros(gens)
        self.avgfit_r = np.zeros(gens)
        self.worstfit_r = np.zeros(gens)
        self.bestind_r = np.zeros(gens)
        self.variance_r = np.zeros(gens)

    def calculateFitness(self):
        fits_c = np.zeros(self.popsize)
        fits_r = np.zeros(self.popsize) 
        for i in range(self.popsize):
            fits = self.fitnessfunction(self.pop_cop[i],self.pop_robber[i])
            fits_c[i] = fits[0]
            fits_r[i] = fits[1]
        return [fits_c,fits_r]

    def run(self):
        # 1 loop for tour
        gen = 0
        for t in range(self.tournaments):
            # 2 pick two to fight (same could be picked -- fix)
            [a,b] = np.random.choice(np.arange(self.popsize),2,replace=False)
            [c,d] = np.random.choice(np.arange(self.popsize),2,replace=False)
            # 3 pick winner
            if self.fit_cop[a] > self.fit_cop[b]:
                winner_c = a
                loser_c = b
            else:
                winner_c = b
                loser_c = a
            if self.fit_robber[c] > self.fit_robber[d]:
                winner_r = c
                loser_r = d
            else:
                winner_r = c
                loser_r = d
            # 4 transfect winner to loser
            for g in range(self.genesize_c):
                if np.random.random() < self.recomprob: 
                    self.pop_cop[loser_c][g] = self.pop_cop[winner_c][g] 
            for g in range(self.genesize_r):
                if np.random.random() < self.recomprob: 
                    self.pop_robber[loser_r][g] = self.pop_robber[winner_r][g] 
            # 5 mutate loser
            self.pop_cop[loser_c] += np.random.normal(0,self.mutationprob,self.genesize_c)
            self.pop_cop[loser_c] = np.clip(self.pop_cop[loser_c],-1,1)
            self.pop_robber[loser_r] += np.random.normal(0,self.mutationprob,self.genesize_r)
            self.pop_robber[loser_r] = np.clip(self.pop_robber[loser_r],-1,1)
            # Update
            fits = self.fitnessfunction(self.pop_cop[loser_c],self.pop_robber[loser_r])
            self.fit_cop[loser_c] = fits[0]
            self.fit_robber[loser_r] = fits[1]
            # 6 Stats 
            if t % self.popsize == 0:
                self.bestfit_c[gen] = np.max(self.fit_cop)
                self.avgfit_c[gen] = np.mean(self.fit_cop)
                self.worstfit_c[gen] = np.min(self.fit_cop)
                self.variance_c[gen] = np.var(self.pop_cop)
                self.bestind_c[gen] = np.argmax(self.fit_cop) 
                
                self.bestfit_r[gen] = np.max(self.fit_robber)
                self.avgfit_r[gen] = np.mean(self.fit_robber)
                self.worstfit_r[gen] = np.min(self.fit_robber)
                self.variance_r[gen] = np.var(self.pop_robber)
                self.bestind_r[gen] = np.argmax(self.fit_robber) 
                gen += 1
#                print(t,np.max(self.fit),np.mean(self.fit),np.min(self.fit),np.argmax(self.fit))

    def showFitness(self):
        plt.plot(self.bestfit_c,label="Best")
        plt.plot(self.avgfit_c,label="Avg.")
        plt.plot(self.worstfit_c,label="Worst")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Evolution Cop")
        plt.show()
        plt.plot(self.variance_c)
        plt.xlabel("Generations")
        plt.ylabel("Variance")
        plt.title("Cop Genotype Variance Over Generations")
        plt.show()
        
        plt.plot(self.bestfit_r,label="Best")
        plt.plot(self.avgfit_r,label="Avg.")
        plt.plot(self.worstfit_r,label="Worst")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Evolution Robber")
        plt.show()
        plt.plot(self.variance_r)
        plt.xlabel("Generations")
        plt.ylabel("Variance")
        plt.title("Robber Genotype Variance Over Generations")
        plt.show()