#%%
import pandas as pd
import spacy
from nltk.corpus import brown
nlp = spacy.load("en_core_web_sm")

#%%
class raised_eyebrow:
    def __init__(self, dictionary, text, corpus_file, tag_category):
        #use the dictionary to map text into word bank
        words = self.map_text(dictionary, text)
        self.tagged_sent = self.pos_tagging(words, tag_category)
        
        # Initialize empty matricies
        self.tags = self.tagged_sent.tags.unique()
        self.words = self.tagged_sent.word
        self.emission_matrix = pd.DataFrame(0, index=self.tags, columns=self.words) 
        self.transition_matrix = pd.DataFrame(0, index=self.tags, columns=self.tags)
        self.initial_probabilities = pd.DataFrame(0, index=self.tags, columns=["Counts"])

        #make the corpus thingy
        self.tags_corpus = pd.read_csv(corpus_file,sep=",",index_col=0)
    
    def map_text(self, dictionary, text):
        f = open(dictionary, "r")
        trans = {}
        lines = f.readlines()
        for line in lines:
            mapping = line.strip().split(", ")
            trans[mapping[0]] = mapping[1]
        # sorted_phrases = sorted(dict.keys(), key=len, reverse=True)
        text = text.replace(":","")
        for original_phrase, new_phrase in trans.items():
            text = text.replace(original_phrase, new_phrase)
        return text
    
    @staticmethod
    def pos_tagging(sentence, category): #returns dataframe with mappings
        rows = []
        for token in nlp(sentence):
            if category == "DEP":
                rows.append((token.text, token.dep_))
            else: rows.append((token.text, token.pos_))
        tags = pd.DataFrame(rows, columns=["word", "tags"])
        return tags
    
    def generate_initial(self):
        for i in self.tags_corpus.index:
            sentence = self.tags_corpus["tags"][i].split(" ")
            if sentence[0] in self.tags: 
                self.initial_probabilities.loc[sentence[0]] += 1
        self.initial_probabilities.Counts = self.initial_probabilities.Counts.div(self.initial_probabilities.Counts.sum(), axis=0)
            
    def generate_hidden(self):
        for i in range(len(self.tagged_sent)):
            self.emission_matrix.loc[self.tagged_sent.loc[i, "tags"], self.tagged_sent.loc[i, "word"]] += 1
        #normalize
        self.emission_matrix = self.emission_matrix.div(self.emission_matrix.sum(axis=1), axis=0)
    
    def training_transition(self): 
        for i in self.tags_corpus.index:
            sentence = self.tags_corpus["tags"][i].split(" ")
            for i in range(len(sentence) - 1): 
                current_tag, next_tag = sentence[i], sentence[i + 1]
                if (current_tag in self.tags) and (next_tag in self.tags):
                    self.transition_matrix.loc[current_tag, next_tag] += 1
        # normalize into probabilities
        self.transition_matrix = self.transition_matrix.div(self.transition_matrix.sum(axis=1), axis=0)

    def translate(self):
        self.generate_initial()
        self.generate_hidden()
        self.training_transition()
        
        hidden = self.emission_matrix
        transition = self.transition_matrix

        #shh don't ask why we didn't add the initial word to the sentence
        state = self.initial_probabilities.Counts.idxmax()
        generated_states = []
        generated_text = ""

        for _ in range(len(self.words)):
            # Pick the state with the highest transition probability from the current state
            state = transition.loc[state].idxmax()
            generated_states.append(state)

            # Pick the observation with the highest emission probability from the current state
            # Check if word has been used, if so remove the column from emission and if sum of row = 0, remove from transition and emission
            word = hidden.loc[state].idxmax()
            generated_text = generated_text + " " + word
            hidden.drop(word, axis="columns", inplace=True)
            if hidden.loc[state].sum() == 0:
               transition.drop(state, axis="columns", inplace=True)
        return generated_states, generated_text
    
   # def run_analysis(self):
        
#%% Run this once!
tags_corpus = pd.DataFrame(columns=["tags"])
sentences = brown.sents()
for sentence in sentences[:10000]:
    tagged_sent = raised_eyebrow.pos_tagging((' '.join(sentence)), "POS")
    tags_corpus.loc[len(tags_corpus.index)] = [' '.join(tagged_sent.tags)]
f = tags_corpus.to_csv("tags_corpus.csv", sep=",")
#%%
# Example usage:
text = "to thine own self be true"
translator = raised_eyebrow("translations.txt", text, "POS_corpus.csv","POS")
#%%
states, sentence = translator.translate()


# %%
