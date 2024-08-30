from util import *

def run():
    text = readTextFromFile("data/shakespeare.txt")
    model = markovifyModel(text)
    poem = generatePoem(model, nrOfLines=4)
    print(poem)
    
    sentiments = analyzeSentiment(poem)
    print(sentiments)
    
    newPoem = replaceWithEmbeddedSynonim(poem)
    print("New poem\n:" + newPoem)
    
    score = scoreBLUE(poem, newPoem)
    print("BLEU score: ", score)

run()
    