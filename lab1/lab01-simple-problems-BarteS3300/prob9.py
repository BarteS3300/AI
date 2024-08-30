import timeit


def sumOfSubMatrix(matrix, x, y):
    """Sum of the elements of a submatrix which starts at coordinates x and ends at coordinates y.

    Args:
        matrix (array): The matrix.
        x (array): The starting coordinates of the submatrix.
        y (array): The ending coordinates of the submatrix.

    Returns:
        int: The sum of the elements of the submatrix.
    """
    sum = 0
    for i in range(x[0], y[0] + 1):
        for j in range(x[1], y[1] + 1):
            sum += matrix[i][j]
    return sum

def sumOfSubMatrixByCopilot(matrix, x, y):
    """
    Same as sumOfSubMatrix but using copilot's suggestion.
    """
    return sum([matrix[i][j] for i in range(x[0], y[0] + 1) for j in range(x[1], y[1] + 1)])

def test():
    assert sumOfSubMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (0, 1), (1, 2)) == 16
    assert sumOfSubMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (0, 1), (0, 1)) == 2
    assert sumOfSubMatrixByCopilot([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (0, 1), (1, 2)) == 16
    assert sumOfSubMatrixByCopilot([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (0, 1), (0, 1)) == 2
    
def main():
    test()
    matrix = [[0, 2, 5, 4, 1], [4, 8, 2, 3, 7], [6, 3, 4, 6, 2], [7, 3, 1, 8, 3], [1, 5, 7, 9, 4]]
    #x = tuple(map(int, input("Enter x1 and x2: ").split()))
    #y = tuple(map(int, input("Enter y1 and y2: ").split()))
    x = [(1, 1), ((2, 2))]
    y = [(3, 3), ((4, 4))]
    start = timeit.default_timer()
    for i in range(len(x)):
        print("The sum of the submatrix is:", sumOfSubMatrix(matrix, x[i], y[i]))
    end = timeit.default_timer()
    print("Time taken by sumOfSubMatrix:", end - start)
    start = timeit.default_timer()
    for i in range(len(x)):
        print("The sum of the submatrix by copilot is:", sumOfSubMatrixByCopilot(matrix, x[i], y[i]), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by sumOfSubMatrixByCopilot:", end - start)
    
main()