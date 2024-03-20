#%%
import pandas as pd
import spacy
from nltk.corpus import brown
nlp = spacy.load("en_core_web_sm")

#%%
class HMM_Translator:
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
    
    def train_initial(self):
        for i in self.tags_corpus.index:
            sentence = self.tags_corpus["tags"][i].split(" ")
            if sentence[0] in self.tags: 
                self.initial_probabilities.loc[sentence[0]] += 1
        #generate probabilities
        self.initial_probabilities.Counts = self.initial_probabilities.Counts.div(self.initial_probabilities.Counts.sum(), axis=0)
            
    def generate_hidden(self):
        for i in range(len(self.tagged_sent)):
            self.emission_matrix.loc[self.tagged_sent.loc[i, "tags"], self.tagged_sent.loc[i, "word"]] += 1
        #generate probabilities
        self.emission_matrix = self.emission_matrix.div(self.emission_matrix.sum(axis=1), axis=0)
    
    def train_transition(self): 
        for i in self.tags_corpus.index:
            sentence = self.tags_corpus["tags"][i].split(" ")
            for i in range(len(sentence) - 1): 
                current_tag, next_tag = sentence[i], sentence[i + 1]
                if (current_tag in self.tags) and (next_tag in self.tags):
                    self.transition_matrix.loc[current_tag, next_tag] += 1
        # normalize into probabilities
        self.transition_matrix = self.transition_matrix.div(self.transition_matrix.sum(axis=1), axis=0)

    def translate(self):
        self.train_initial()
        self.generate_hidden()
        self.train_transition()
        
        hidden = self.emission_matrix.copy()
        transition = self.transition_matrix.copy()

        #NOTE: from here, stuff gets kinda messy as I adjusted the code to get data for the presentation
        #so if youre trying to run this, you may have to debug
        state = self.initial_probabilities.Counts.idxmax()
        generated_states = [state]
        generated_text = ""
        probability = self.initial_probabilities.Counts[state]
        word = hidden.loc[state].idxmax()
        generated_text = generated_text + " " + word
        hidden.drop(word, axis="columns", inplace=True)
        if hidden.loc[state].sum() == 0:
            transition.drop(state, axis="columns", inplace=True)

        for _ in range(len(self.words)-1):
            # Pick the state with the highest transition probability from the current state
            prev_state = state
            state = transition.loc[state].idxmax()
            generated_states.append(state)
            probability *= transition.loc[prev_state][state]

            # Pick the observation with the highest emission probability from the current state
            # Check if word has been used, if so remove the column from emission and if sum of row = 0, remove from transition and emission
            word = hidden.loc[state].idxmax()
            generated_text = generated_text + " " + word
            hidden.drop(word, axis="columns", inplace=True)
            if hidden.loc[state].sum() == 0:
               transition.drop(state, axis="columns", inplace=True)
        return generated_states, generated_text, probability
    
   # def run_analysis(self):
        
#%% Run this once!
tags_corpus = pd.DataFrame(columns=["tags"])
sentences = brown.sents()
for sentence in sentences[:10000]:
    tagged_sent = HMM_Translator.pos_tagging((' '.join(sentence)), "POS")
    tags_corpus.loc[len(tags_corpus.index)] = [' '.join(tagged_sent.tags)]
f = tags_corpus.to_csv("POS_corpus.csv", sep=",")
#%%
# Example usage:
text = "to thine own self be true"
translator = HMM_Translator("translations.txt", text, "POS_corpus.csv","POS")
#%%
states, sentence, probability = translator.translate()
print(states, sentence, probability)

# %%DEP timeee
tags_corpus = pd.DataFrame(columns=["tags"])
sentences = brown.sents()
for sentence in sentences[:10000]:
    tagged_sent = HMM_Translator.pos_tagging((' '.join(sentence)), "DEP")
    tags_corpus.loc[len(tags_corpus.index)] = [' '.join(tagged_sent.tags)]
f = tags_corpus.to_csv("DEP_corpus.csv", sep=",")
# %%
dep_translator = HMM_Translator("translations.txt", text, "DEP_corpus.csv","DEP")
dstates, dsentence, dprobability = dep_translator.translate()
print(dstates, dsentence, str(dprobability*100) + "% ")
# %%
