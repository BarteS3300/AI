from random import uniform

def generateNewValue(lim1, lim2):
    return int(uniform(lim1, lim2))

def binToInt(x):
    val = 0
    # x.reverse()
    for bit in x:
        val = val * 2 + bit
    return val

def normalizeList(l):
    c = 0
    sortedSetL = sorted(set(l))
    for i in sortedSetL:
        for j in range(len(l)):
            if l[j] == i:
                l[j] = c
        c += 1
    return l

