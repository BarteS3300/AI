from util import *

def run():
    text = readTextFromFile("data/proverbe.txt")
    marcov_chain = generateMarcovChain(text, states=2)
    print(marcov_chain)
    new_text = generate_text(marcov_chain, lenght=200)
    print(new_text)
    
run()