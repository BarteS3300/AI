from random import randint, uniform
from fcOptimisation.utils import generateNewValue, normalizeList
import networkx as nx

class NetworkChromosome:
    def __init__(self, problParam = None):
        self.__problParam = problParam
        self.initilizeChromosome()
        self.__fitness = 0.0
    
    @property
    def repres(self):
        return self.__repres
    
    @property
    def fitness(self):
        return self.__fitness 
    
    @repres.setter
    def repres(self, l = []):
        self.__repres = l 
    
    @fitness.setter 
    def fitness(self, fit = 0.0):
        self.__fitness = fit 
    
    def initilizeChromosome(self):
        self.__repres = [-1] * len(self.__problParam['network'].nodes)
        
        community = 0
        
        
        # nodes = self.__problParam['network'].nodes()
        # edges = self.__problParam['network'].number_of_edges()
        
        mat = nx.to_numpy_array(self.__problParam['network'])
        neighbours = {}
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 1:
                    neighbours.setdefault(i, []).append(j)

        sortedNeighbours = {k: v for k, v in sorted(neighbours.items(), key=lambda item: len(item[1]), reverse=True)}
        # print("Sorted", sortedNeighbours)
        
        for key in sortedNeighbours.keys():
            if self.__repres[key] == -1:
                self.__repres[key] = community
                for neighbour in sortedNeighbours[key]:
                    if self.__repres[neighbour] == -1:
                        self.__repres[neighbour] = community
                community += 1
        # print("Repres", self.__repres)
    
    def crossover(self, c):
        r = randint(0, len(self.__repres) - 1)
        newrepres = []
        for i in range(r):
            newrepres.append(self.__repres[i])
        for i in range(r, len(self.__repres)):
            newrepres.append(c.__repres[i])
        offspring = NetworkChromosome(c.__problParam)
        offspring.repres = newrepres
        return offspring
    
    def mutation(self):
        mat = nx.to_numpy_array(self.__problParam['network'])
        while True:
            pos = randint(0, len(self.__repres) - 1)
            validNeighbours = []
            for i in range(len(mat[pos])):
                if mat[pos][i] == 1 and self.__repres[i] != self.__repres[pos]:
                    validNeighbours.append(self.__repres[i])
            if(len(validNeighbours) > 0):
                break
        self.__repres[pos] = validNeighbours[randint(0, len(validNeighbours) - 1)]
        # self.__repres[pos] = generateNewValue(0, max(self.__repres))
        self.__repres = self.__repres
        
    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness

# c = NetworkChromosome({'network': nx.read_gml('communityDetection/real/krebs/krebs.gml')})
# c1 = NetworkChromosome({'network': nx.read_gml('communityDetection/real/krebs/krebs.gml')})