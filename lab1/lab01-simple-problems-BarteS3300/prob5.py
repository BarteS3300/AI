import timeit

def repetedValue(n):
    """This function receives a list of n numbers which contains values from (1, 2, ..., n-1) with only one value repeted and returns the repeated value in the list.

    Args:
        n (array): The list of numbers.

    Returns:
        int: The repeated value in the list.
    """
    return int(len(n) - (len(n) * (len(n) + 1) / 2 - sum(n)))

def repetedValueByCopilot(n):
    """
    Same as repetedValue but using copilot's suggestion.
    """
    return sum(n) - sum(set(n))
    

def test():
    assert repetedValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]) == 10
    assert repetedValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5]) == 5
    assert repetedValue([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]) == 1
    assert repetedValueByCopilot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]) == 10
    assert repetedValueByCopilot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5]) == 5
    assert repetedValueByCopilot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]) == 1
    

def main():
    test()
    n = list(map(int, input("Enter a list of numbers: ").split()))
    start = timeit.default_timer()
    print("The repeated value is:", repetedValue(n))
    end = timeit.default_timer()
    print("Time taken by repetedValue:", end - start)
    start = timeit.default_timer()
    print("The repeated value is:", repetedValueByCopilot(n), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by repetedValueByCopilot:", end - start)
    
main()