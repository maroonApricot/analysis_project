#%%
import pandas as pd
import spacy
from nltk.corpus import brown
nlp = spacy.load("en_core_web_sm")
#%% Run once!
corpus = pd.DataFrame(columns=["words"])
sentences = brown.sents()
for sentence in sentences[:100000]:
    corpus.loc[len(corpus.index)] = [' '.join(sentence)]
f = corpus.to_csv("word_corpus.csv", sep=",")
#%% then run this
corpus = pd.read_csv("word_corpus.csv",sep=",",index_col=0)
#%%
text = "to your own self be true"
words = text.split(sep=" ")
#%%
transition_matrix = pd.DataFrame(0, index=words, columns=words)
initial_probabilities = pd.DataFrame(0, index=words, columns=["Counts"])

# initial states + probabilities
for i in corpus.index:
    sentence = corpus["words"][i].split(" ")
    if sentence[0] in words: 
        initial_probabilities.loc[sentence[0]] += 1
#generate probabilities
initial_probabilities.Counts = initial_probabilities.Counts.div(initial_probabilities.Counts.sum(), axis=0)

#transition states + probabilities
for i in corpus.index:
    sentence = corpus["words"][i].split(" ")
    for i in range(len(sentence) - 1): 
        current_word, next_word = sentence[i], sentence[i + 1]
        if (current_word in words) and (next_word in words):
            transition_matrix.loc[current_word, next_word] += 1
# normalize into probabilities
transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
#Account for NaN (basically no data --> division by zero):


#generate most probable sequence
transition = transition_matrix.copy()

word = initial_probabilities.Counts.idxmax()
generated_text = word
probability = initial_probabilities.Counts[word]
transition.drop(word, axis="columns", inplace=True)

for _ in range(len(words)-1):
    prev_word = word
    word = transition.loc[word].idxmax()
    generated_text = generated_text + " " + str(word)
    print(transition.loc[prev_word][word])
    probability *= transition.loc[prev_word][word]
    transition.drop(word, axis="columns", inplace=True)

print(generated_text, str(probability*100) + "% ")
# %%
