from collections import Counter
import timeit

def wordsWhatAppearOnce(s):
    """This function receives a string and returns a list of words that appear only once in the string.

    Args:
        s (string): The string to be processed.

    Returns:
        array: The list of words that appear only once in the string.
    """
    words = s.split()
    wordsCount = {}
    for word in words:
        wordsCount[word] = wordsCount.get(word, 0) + 1
    return [word for word in wordsCount if wordsCount[word] == 1]

def wordsWhatAppearOnceByCopilot(s):
    """
    Same as wordsWhatAppearOnce but using copilot's suggestion.
    """
    word_counts = Counter(s.split())
    return [word for word, count in word_counts.items() if count == 1]

def test():
    assert(wordsWhatAppearOnce("I am a student and I am a teacher and I am a Teacher") == ['student', 'teacher', 'Teacher'])
    assert(wordsWhatAppearOnce("I am a student and I am a teacher and I am a teacher") == ['student'])
    assert(wordsWhatAppearOnce("I am a student and I am a Teacher and") == ['student', 'Teacher'])
    assert(wordsWhatAppearOnceByCopilot("I am a student and I am a teacher and I am a Teacher") == ['student', 'teacher', 'Teacher'])
    assert(wordsWhatAppearOnceByCopilot("I am a student and I am a teacher and I am a teacher") == ['student'])
    assert(wordsWhatAppearOnceByCopilot("I am a student and I am a Teacher and") == ['student', 'Teacher'])
    

def main():
    test()
    s = input("Enter a string: ")
    start = timeit.default_timer()
    print("The words that appear only once are: ", wordsWhatAppearOnce(s))
    end = timeit.default_timer()
    print("Time taken by wordsWhatAppearOnce:", end - start)
    start = timeit.default_timer()
    print("The words that appear only once are by Copilot: ", wordsWhatAppearOnceByCopilot(s), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by wordsWhatAppearOnceByCopilot:", end - start)
    
main()