import timeit


def colourZone(m, i, j, zone):
    """Change the 0s in the zone to the zone number.

    Args:
        m (array): The matrix.
        i (int): The current row of the zone.
        j (int): The current column of the zone.
        zone (int): The zone number.
    """
    neighbours = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    for x, y in neighbours:
        if -1 < y < len(m[0]) and -1 < x < len(m) and m[x][y] == 0:
            m[x][y] = zone
            colourZone(m, x, y, zone)


def decolorateZone(m, notSurroundedBy1):
    """Change the zones that are not surrounded by 1s to 0s and the zones that are surrounded by 1s to 1s.

    Args:
        m (array): The matrix.
        notSurroundedBy1 (array): The zones that are not surrounded by 1s.
    """
    for i in range(0,len(m)):
        for j in range(0, len(m[i])):
            if m[i][j] in notSurroundedBy1:
                m[i][j] = 0
            else:
                m[i][j] = 1

def findZones(m):
    """Find the zones in the matrix and change the zones of 0s that are surrounded by 1s in 1s.

    Args:
        m (array): The matrix of 0s and 1s.

    Returns:
        array: The matrix with the zones of 0s that are surrounded by 1s changed in 1s.
    """
    zona = 2
    for i in range(0,len(m)):
        for j in range(0, len(m[i])):
            if m[i][j] == 0:
                m[i][j] = zona
                colourZone(m, i, j, zona)
                zona += 1

    notSurroundedBy1 = set()
    
    for j in range(0, len(m[0])):
        if m[0][j] != 1:
            notSurroundedBy1.add(m[0][j])
        if m[len(m) - 1][j] != 1 and m[len(m) - 1][j] not in notSurroundedBy1:
            notSurroundedBy1.add(m[len(m) - 1][j])
    for i in range(0, len(m)):
        if m[i][0] != 1:
            notSurroundedBy1.add(m[i][0])
        if m[i][len(m[i]) - 1] != 1:
            notSurroundedBy1.add(m[i][len(m[i]) - 1])
            
    decolorateZone(m, notSurroundedBy1)
    return m

def printMatrix(m):
    """
    Print a matrix.
    """
    for i in m:
        print(i)
        

def test():
    assert(findZones([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert(findZones([[1, 0, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 0, 1], [1, 0, 1], [1, 1, 1]])

def main():
    test()
    m = [[1,1,1,1,0,0,1,1,0,1], [1,0,0,1,1,0,1,1,1,1], [1,0,0,1,1,1,1,1,1,1], [1,1,1,1,0,0,1,1,0,1], [1,0,0,1,1,0,1,1,0,0], [1,1,0,1,1,0,0,1,0,1], [1,1,1,0,1,0,1,0,0,1], [1,1,1,0,1,1,1,1,1,1]]
    start = timeit.default_timer()
    printMatrix(findZones(m))
    end = timeit.default_timer()
    print("Time taken by findZones:", end - start)
    printMatrix(findZones(m))

main()