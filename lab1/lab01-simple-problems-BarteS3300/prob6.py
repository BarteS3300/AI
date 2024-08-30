import timeit


def majorityElement(n):
    """This function receives a list of n numbers and returns the majority element in the list.

    Args:
        n (array): The list of numbers.

    Returns:
        int: The majority element in the list.
    """
    return max(set(n), key = n.count)

def majorityElementByCopilot(n):
    """
    Same as majorityElement but using copilot's suggestion.
    """
    contor = 0
    candidat = None

    for num in n:
        if contor == 0:
            candidat = num
        contor += (1 if num == candidat else -1)

    return candidat
    

def main():
    n = list(map(int, input("Enter a list of numbers: ").split()))
    start = timeit.default_timer()
    print("The majority element is:", majorityElement(n))
    end = timeit.default_timer()
    print("Time taken by majorityElement:", end - start)
    start = timeit.default_timer()
    print("The majority element is:", majorityElementByCopilot(n), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by majorityElementByCopilot:", end - start)
    
main()