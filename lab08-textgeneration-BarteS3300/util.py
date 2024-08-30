import random
import markovify

def readTextFromFile(path, states=1):
    with open(path, 'r') as file:
        return file.read()

def generateMarcovChain(text, states=1):
    words = text.split(" ")
    markov_chain = {}

    for i in range(len(words) - states):
        key = ""
        for j in range(states):
            key += words[i + j] + " "
        key = key[:-1]
        if key in markov_chain:
            markov_chain[key].append(words[i + states])
        else:
            markov_chain[key] = [words[i + states]]

    return markov_chain

def generate_text(marcov_chain, lenght = 200):
    text = ""
    states = len(list(marcov_chain.keys())[0].split(" "))
    words = random.choice(list(marcov_chain.keys()))
    text += words
    for i in range(lenght):
        if words in marcov_chain:
            next_word = random.choice(marcov_chain[words])
            text += " " + next_word
            words = " ".join(text.split(" ")[-states:])
        else:
            break
    return text

def markovifyModel(text):
    return markovify.Text(text)

def generatePoem(model, nrOfLines = 4):
    poem = ""
    for i in range(nrOfLines):
        poem += model.make_short_sentence(100) + "\n"
    return poem

def analyzeSentiment(text):
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    sentiments = {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    lines = text.split("\n")
    sia = SentimentIntensityAnalyzer()
    for line in lines:
        sentiment_tool = sia.polarity_scores(line)
    
        for sentimen in sentiments:
            sentiments[sentimen] += sentiment_tool[sentimen]
    
    sentiments = {key: value / len(lines) for key, value in sentiments.items()}
    
    return sentiments

def replaceWithEmbeddedSynonim(poem):
    import spacy

    nlp = spacy.load('en_core_web_md')

    tokens = ""
    text = readTextFromFile("data/shakespeare.txt")
    tokens = nlp(text)


    newPoem = ""
    for word in poem.split():
        token1 = nlp(word)
        
        maxSimilarity = -1
        mostSimilarWord = word
        
        for token2 in tokens:
            similarity = token1.similarity(token2)
            
            if similarity > maxSimilarity:
                maxSimilarity = similarity
                mostSimilarWord = token2.text
        
        newPoem += mostSimilarWord + " "

        if mostSimilarWord.endswith('.') or mostSimilarWord.endswith('?') or mostSimilarWord.endswith('!'):
            newPoem += "\n"

    return newPoem

def scoreBLUE(poem, newPoem):
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import sentence_bleu

    import sacrebleu

    reference = [' '.join(word_tokenize(poem))]
    candidate = [' '.join(word_tokenize(newPoem))]
    
    score = sacrebleu.raw_corpus_bleu(candidate, [reference], .01).score

    return score