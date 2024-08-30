from math import sqrt
import math
import timeit

def distance(x, y):
    """The Euclidean distance between two points x and y.

    Args:
        x (tuple): The first point which is a tuple of two numbers.
        y (tuple): The second point which is a tuple of two numbers.
        
    Returns:
        float: The distance between the two points.
    """

    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    #return math.dist(x, y)

def distanceByCopilot(x, y):
    """
    Same as distance but using copilot's suggestion.
    """
    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(x, y)))
    
def test():
    assert distance((0, 0), (0, 0)) == 0
    assert distance((0, 0), (1, 0)) == 1
    assert distance((1, 5), (4, 1)) == 5
    assert distanceByCopilot((0, 0), (0, 0)) == 0
    assert distanceByCopilot((0, 0), (1, 0)) == 1
    assert distanceByCopilot((1, 5), (4, 1)) == 5

def main():
    test()
    x1 = float(input("Enter x1: "))
    x2 = float(input("Enter x2: "))
    y1 = float(input("Enter y1: "))
    y2 = float(input("Enter y2: "))
    x = (x1, x2)
    y = (y1, y2)
    start = timeit.default_timer()
    print("The distance between the points is:", distance(x, y))
    end = timeit.default_timer()
    print("Time taken by distance:", end - start)
    start = timeit.default_timer()
    print("The distance between the points is:", distanceByCopilot(x, y), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by distanceByCopilot:", end - start)
    
main()