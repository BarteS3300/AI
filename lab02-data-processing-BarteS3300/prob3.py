import re
import unidecode
import translators as ts
from nltk.corpus import wordnet
import nltk

def readFromTextFile(filename):
    text = ''
    try:
        f = open(filename, 'r', encoding='utf-8')
        text = f.read()
        f.close()
    except IOError:
        print("Fisier corupt!")
    return text

def numberOfSentences(filename):
    text = readFromTextFile(filename)
    return len(re.split(r'[.!?:]', text))

def wordsFromText(filename):
    text = readFromTextFile(filename)
    words = filter(None, re.split(r'[ \n!;:?.,"”]', text))
    return (list(words))

def numberOfWords(filename):
    return len(wordsFromText(filename))

def diferentWords(filename):
    return len(set(wordsFromText(filename)))

def findShortestAndLargestWords(filename):
    diferentWords = set(wordsFromText(filename))
    sortedWords = sorted(diferentWords, key=len)
    shortestWords = [word for word in sortedWords if len(word) == len(sortedWords[0])]
    longestWords = [word for word in sortedWords if len(word) == len(sortedWords[-1])]
    return [shortestWords, longestWords]

def replaceDiacritics(filename):
    return unidecode.unidecode(readFromTextFile(filename))

def findSynonymsForThrLongestWord(filename):
    longestWords = findShortestAndLargestWords(filename)[1]
    synonyms = {}
    for word in longestWords:
        synonyms[word] = []
        wordInEnglish = ts.translate_text(word, from_language='ro' ,to_language='en')
        for syn in wordnet.synsets(wordInEnglish):
            synonyms[word].append(ts.translate_text(syn.lemmas()[0].name().replace("_"," "), from_language='en' ,to_language='ro'))
    print(synonyms)
        
def main():
    filename = "texts.txt"
    #print(numberOfSentences(filename))
    #print(numberOfWords(filename))
    #print(diferentWords(filename))
    #print("The shortest words are:", findShortestAndLargestWords(filename)[0], "and the largest words are:", findShortestAndLargestWords(filename)[1])
    #print(replaceDiacritics(filename))
    findSynonymsForThrLongestWord(filename)
    
main()