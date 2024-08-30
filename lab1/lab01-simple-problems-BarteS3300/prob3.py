import timeit


def produsScalar(a, b):
    """This function receives two vectors of any dimensopn and returns their scalar product.

    Args:
        a (array): The first vector.
        b (array): The second vector.

    Returns:
        int: The scalar product of the two vectors.
    """
    if not hasattr(a, '__len__'):
        return a * b
    else:
        return sum(produsScalar(x, y) for x, y in zip(a, b))
    
def produsScalarByCopilot(a, b):
    """
    Same as produsScalar but using copilot's suggestion.
    """
    if isinstance(a[0], list):
        return sum(produsScalarByCopilot(sub_a, sub_b) for sub_a, sub_b in zip(a, b))
    else:
        return sum(x*y for x, y in zip(a, b))
    
def test():
    assert produsScalar([1, 2, 3], [1, 2, 3]) == 14
    assert produsScalarByCopilot([1, 2, 3], [1, 2, 3]) == 14
    assert produsScalar([[1,0,2,0,3], [1,2,0,3,1]], [[1,2,0,3,1], [1,0,2,0,3]]) == 8
    assert produsScalarByCopilot([[1,0,2,0,3], [1,2,0,3,1]], [[1,2,0,3,1], [1,0,2,0,3]]) == 8
    
    
def main():
    test()
    a =  [[1,0,2,0,3], [1,2,0,3,1]]
    b = [[1,2,0,3,1], [1,0,2,0,3]]
    start = timeit.default_timer()
    print("Scalar product of the two vectors is:", produsScalar(a, b))
    end = timeit.default_timer()
    print("Time taken by produsScalar:", end - start)
    start = timeit.default_timer()
    print("Scalar product of the two vectors using produsScalarByCopilot is:", produsScalarByCopilot(a, b))
    end = timeit.default_timer()
    print("Time taken by produsScalarByCopilot:", end - start)
    
main()