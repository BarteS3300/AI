import timeit


def findRowIndex(a):
    """Find the row index with the most 1s in a matrix.

    Args:
        a (matrix): The matrix to be processed.

    Returns:
        int: The index of the row with the most 1s.
    """
    index = 0
    pos1 = len(a[0]) - 1
    for row in a:
        if row[pos1] == 1 and row[pos1 - 1] == 1:
            s = 0
            m = (s + pos1) // 2
            while s < pos1:
                if row[m] == 1 and row[m - 1] == 0:
                    index = a.index(row)
                    break
                elif row[m] == 0:
                    s = m + 1
                else:
                    pos1 = m
                m = (s + pos1) // 2
    return index       


def findRowIndexByCopilot(a):
    """
    Same as findRowIndex but using copilot's suggestion.
    """
    index_max = max(range(len(a)), key=lambda index: sum(a[index]))
    return index_max

def test():
    assert findRowIndex([[0,0,0,1,1], [0,1,1,1,1], [0,0,1,1,1]]) == 1
    assert findRowIndex([[0,0,0,1,1], [0,1,1,1,1], [0,1,1,1,1]]) == 1
    assert findRowIndex([[1,1,1,1,1], [0,1,1,1,1], [0,0,1,1,1]]) == 0
    assert findRowIndexByCopilot([[0,0,0,1,1], [0,1,1,1,1], [0,0,1,1,1]]) == 1
    assert findRowIndexByCopilot([[0,0,0,1,1], [0,1,1,1,1], [0,1,1,1,1]]) == 1
    assert findRowIndexByCopilot([[1,1,1,1,1], [0,1,1,1,1], [0,0,1,1,1]]) == 0
    
def main():
    test()
    a = [[0,0,0,1,1], [0,1,1,1,1], [0,0,1,1,1]]
    start = timeit.default_timer()
    print("Linia", findRowIndex(a) + 1, "contine cei mai multi de 1.")
    end = timeit.default_timer()
    print("Time taken by findRowIndex:", end - start)
    start = timeit.default_timer()
    print("Linia", findRowIndexByCopilot(a) + 1, "contine cei mai multi de 1. (copilot)")
    end = timeit.default_timer()
    print("Time taken by findRowIndexByCopilot:", end - start)
    
main()