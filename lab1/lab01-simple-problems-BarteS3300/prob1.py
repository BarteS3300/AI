import timeit

def lastWordByAlphabetically(s):
    """Return the last word(by alphabet) in the string s.

    Args:
        s (string): The string to be processed.

    Returns:
        string: The last word of the string alphabetically.
    """
    lastWord = ""
    words = s.split()
    for word in words:
        if word.lower() > lastWord.lower():
            lastWord = word
    return lastWord

def lastWordByAlphabeticallyByCopilot(s):
    """
    Same as lastWordByAlphabetically but using copilot's suggestion.
    """
    words = s.split()
    return max(words, key=str.lower)

def test():
    assert(lastWordByAlphabetically("I am a student") == "student")
    assert(lastWordByAlphabetically("I am a student and I am a teacher") == "teacher")
    assert(lastWordByAlphabetically("I am a student and I am a Teacher") == "Teacher")
    assert(lastWordByAlphabetically("I am a student and I am a teacher and I am a teacher") == "teacher")
    assert(lastWordByAlphabetically("I am a student and I am a teacher and I am a Teacher") == "teacher")
    assert(lastWordByAlphabetically("I am a student and I am a Teacher and I am a teacher") == "Teacher")
    assert(lastWordByAlphabetically("I am a student and I am a TEACHER and I am a Teacher") == "TEACHER")
    assert(lastWordByAlphabeticallyByCopilot("I am a student") == "student")
    assert(lastWordByAlphabeticallyByCopilot("I am a student and I am a teacher") == "teacher")
    assert(lastWordByAlphabeticallyByCopilot("I am a student and I am a Teacher") == "Teacher")
    assert(lastWordByAlphabeticallyByCopilot("I am a student and I am a teacher and I am a teacher") == "teacher")
    assert(lastWordByAlphabeticallyByCopilot("I am a student and I am a teacher and I am a Teacher") == "teacher")
    assert(lastWordByAlphabeticallyByCopilot("I am a student and I am a Teacher and I am a teacher") == "Teacher")
    assert(lastWordByAlphabeticallyByCopilot("I am a student and I am a TEACHER and I am a Teacher") == "TEACHER")
    

def main():
    test()
    #s = input("Enter a string: ")
    s = "I am a student and I am a teacher and I am a Teacher"
    start = timeit.default_timer()
    print("The last word by alphabetically is:", lastWordByAlphabetically(s))
    end = timeit.default_timer()
    print("Time taken by lastWordByAlphabetically:", end - start)
    start = timeit.default_timer()
    print("The last word by alphabetically is:", lastWordByAlphabeticallyByCopilot(s), " (copilot)")
    end = timeit.default_timer()
    print("Time taken by lastWordByAlphabeticallyByCopilot:", end - start)

main()