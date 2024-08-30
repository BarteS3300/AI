import timeit


def firstnNumbersInBinary(n):
    """This function receives a number n and returns a list of the first n numbers in binary.

    Args:
        n (int): The number of numbers to be converted to binary.

    Returns:
        list: The list of the first n numbers in binary.
    """
    return [bin(i)[2:] for i in range(1, n+1)]

def firstnNumbersInBinaryByCopilot(n):
    """
    Same as firstnNumbersInBinary but using copilot's suggestion.
    """
    return [f'{i:b}' for i in range(1, n+1)]

def test():
    assert firstnNumbersInBinary(5) == ['1', '10', '11', '100', '101']
    assert firstnNumbersInBinaryByCopilot(5) == ['1', '10', '11', '100', '101']
    
def main():
    test()
    n = int(input("Enter a number: "))
    start = timeit.default_timer()
    print("The first", n, "numbers in binary are:", firstnNumbersInBinary(n))
    end = timeit.default_timer()
    print("Time taken by firstnNumbersInBinary:", end - start)
    start = timeit.default_timer()
    print("The first", n, "numbers in binary are:", firstnNumbersInBinaryByCopilot(n), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by firstnNumbersInBinaryByCopilot:", end - start)
    
main()
    