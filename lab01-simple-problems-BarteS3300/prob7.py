import timeit


def kHighestNumber(n, k):
    """This function receives a list of n numbers and returns the kth highest number in the list.

    Args:
        n (array): The list of numbers.
        k (int): The kth highest number to be found.

    Returns:
        int: The kth highest number in the list.
    """
    return sorted(n)[-k]

def kHighestNumberByCopilot(n, k):
    """
    Same as kHighestNumber but using copilot's suggestion.
    """
    return sorted(n, reverse=True)[k-1]

def test():
    assert kHighestNumber([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3) == 8
    assert kHighestNumber([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1) == 10
    assert kHighestNumber([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10) == 1
    assert kHighestNumberByCopilot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3) == 8
    assert kHighestNumberByCopilot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1) == 10
    assert kHighestNumberByCopilot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10) == 1

def main():
    test()
    n = list(map(int, input("Enter a list of numbers: ").split()))
    k = int(input("Enter k: "))
    start = timeit.default_timer()
    print("The kth highest number is:", kHighestNumber(n, k))
    end = timeit.default_timer()
    print("Time taken by kHighestNumber:", end - start)
    start = timeit.default_timer()
    print("The kth highest number is:", kHighestNumberByCopilot(n, k), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by kHighestNumberByCopilot:", end - start)
    
main()