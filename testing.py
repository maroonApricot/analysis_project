import spacy
import random
nlp = spacy.load("en_core_web_sm")

# add more stuff in here if you want
word_mapping = {
    "thine": "your",
}
shakespeare_sentence = "To thine own self be true"

tokens = nlp(shakespeare_sentence)
modern_tokens = [word_mapping.get(token.text.lower(), token.text) for token in tokens]
pos_tags = [token.pos_ for token in tokens]

pos_transitions = {
    'ADJ': ['NOUN'],  
    'NOUN': ['VERB'], 
    'VERB': ['ADP'],  
    'ADP': ['ADJ', 'NOUN'],  
}
def reorder_sentence(tokens, pos_tags):
    ordered_tokens = []
    for pos in pos_tags:
        next_pos_options = pos_transitions.get(pos, None)
        if next_pos_options:
            next_pos = random.choice(next_pos_options)
            for i, tag in enumerate(pos_tags):
                if tag == next_pos and tokens[i] not in ordered_tokens:
                    ordered_tokens.append(tokens[i])
                    break
    return ordered_tokens

reordered_tokens = reorder_sentence(modern_tokens, pos_tags)
reordered_sentence = " ".join(reordered_tokens)

print("Shakespearean Sentence:", shakespeare_sentence)
print("Modern English Translation:", " ".join(modern_tokens))
print("Reordered Modern English Sentence (Simplified Markov Chain):", reordered_sentence)
