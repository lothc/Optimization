"""
Python implementation of a genetic algorithm 
by Christophe Loth - May 2012

Features a "speedy" vectorized genetic algorithm calculation as 
proposed in Mitchell (1996)

"""
from __future__ import division
#import matplotlib
import numpy as np
import sys

class geneticAlgorithm:

    def __init__(self,genLength,sizePop,numGenMax=1000,crossoverProb=1,mutationProb=0.003,crossoverType=2,useMaskFlag=1):
        self.genLength=genLength # Genome length
        self.sizePop=sizePop # Population size
        self.numGenMax=numGenMax # Maximum number of generations in a single run
        self.crossoverProb=crossoverProb # Crossover probability
        self.mutationProb=mutationProb # Bitwise mutation probability
        self.crossoverType=crossoverType # 0 <=> no crossover; 1 <=> 1pt crossover; 2 <=> uniform crossover
        self.useMaskFlag=useMaskFlag # indicates whether to use precomputed masks for the mutation process
    
    
    def fitEval(self,pop):
        return np.sum(pop, axis=1)
            
    def __call__(self):
        # mask used when no crossover
        noCrossmasks=np.zeros((self.sizePop, self.genLength), dtype=bool)
        
        # masks used when crossover
        maskFactor=5
        uniformCrossmask=np.random.rand(self.sizePop/2,(self.genLength+1)*maskFactor)<0.5 
        mutmask=np.random.rand(self.sizePop,(self.genLength+1)*maskFactor)<self.mutationProb 

        # print fitness values
        avgFitHist=np.zeros((1,self.numGenMax+1))
        maxFitHist=np.zeros((1,self.numGenMax+1)) 


        eliteInd=[] 
        eliteFit=-sys.float_info.max 


        # the population is a sizePop by genLength matrix of random booleans
        pop=np.random.rand(self.sizePop,self.genLength)<0.5 

        for gen in range(self.numGenMax):
            # compute the population fitness. 
            # The vector of fitness values should be of size sizePop.
            fitVals=  self.fitEval(pop)
            
            maxFitHist[0,gen+1]=max(fitVals)
            maxIndex=np.argmax(fitVals)
            
            avgFitHist[0,gen+1]=np.mean(fitVals) 
            if eliteFit<maxFitHist[0,gen+1]:
                eliteFit=maxFitHist[0,gen+1] 
                eliteInd=pop[maxIndex,:] 
    

            print('generation = '+str(gen+1)+', avgFitness = '+ str(avgFitHist[0,gen+1])+ ', maxFitness = '+ str(maxFitHist[0,gen+1]))
 
            # Sigma scaling 
            sigma=np.std(fitVals) 
            if sigma!=0:
                fitVals=1+(fitVals-np.mean(fitVals))/sigma
                fitVals[fitVals<=0]=0 
            else:
                fitVals=np.ones((self.sizePop,1)) 
            
            
            # Computes cumulative normalized fitness values 
            cumNormFitVals=np.cumsum(fitVals/sum(fitVals)) 

            
            ltemp=range(1,(self.sizePop+1))
            ltemp[:] = [x / self.sizePop for x in ltemp]
            markers=np.random.rand(1,1)+ltemp
            
            markers[markers>1]=markers[markers>1]-1 
            
            y=cumNormFitVals
            np.insert(y,0,0)

            parentInd=np.digitize(markers[0,:],y)
            parentInd=parentInd[np.random.permutation(self.sizePop)]     
 
            # first parents from each pair
            parents1=pop[parentInd[0:self.sizePop/2],:] 
            
            # second parents from each pair
            parents2=pop[parentInd[self.sizePop/2:],:] 
     
            # crossover masks
            if self.crossoverType==0:
                masks=noCrossmasks 
            elif self.crossoverType==1:
                masks=np.zeros((self.sizePop/2, self.genLength), dtype=bool)
                temp=np.ceil(np.random.rand(self.sizePop/2,1)*(self.genLength-1)) 
                for i in range(1,self.sizePop/2):
                    masks[i,1:temp[i]]=True 
            else:
                if self.useMaskFlag:
                    temp=int(np.random.rand(1)*self.genLength*(maskFactor-1)) 
                    masks=uniformCrossmask[:,temp+1:temp+self.genLength] 
                else:
                    masks=np.random.rand(self.sizePop/2, self.genLength)<.5 

     
            # uncrossed parent pairs:
            reprodInd=np.random.rand(self.sizePop/2,1)<(1-self.crossoverProb)
            masks[reprodInd[:,0],:]=False 
            
            # crossover process
            kids1=parents1 
            kids1[masks]=parents2[masks] 
            kids2=parents2 
            kids2[masks]=parents1[masks] 
            
            pop=np.concatenate((kids1,kids2)) 
            
            # mutation process
            if self.useMaskFlag:
                n0=int(np.random.rand(1)*self.genLength*(maskFactor-1)) 
                masks=mutmask[:,(n0):(n0+self.genLength)] 
            else:
                masks=np.random.rand(self.sizePop, self.genLength)<self.mutationProb 
            
            pop=np.logical_xor(pop,masks) 
               
        return pop 
 

# example:
                   
s=geneticAlgorithm(100,500)
print(s())
     
