#!/usr/bin/env python
# coding: utf-8

# # Answers
# 
# ## Task 1
# Threshold: 2
# 
# Total size of vocabulary: 23192
# 
# Occurences of \<unk>: 5449
#     
# Occurences of \<unk-num>: 267
#     
# Occurences of \<unk-all-caps>: 663
#     
# Occurences of \<unk-upper>: 5498
#     
# Occurences of \<unk-punc>: 6159
#     
# Occurences of \<unk-noun>: 1076
#     
# Occurences of \<unk-verb>: 142
#     
# Occurences of \<unk-adj>: 746
#     
# Occurences of \<unk-adv>: 11
# 
# ## Task 2
# 
# Transition Parameters in HMM: 1416
# 
# Emission Parameters in HMM: 30389
# 
# ## Task 3
# 
# Greedy Decoding Accuracy: 0.944
# 
# ## Task 4
# 
# Viterbi Decoding Accuracy: 0.915
# 
# ## Explanation
# For the vocabulary creation, I used a threshold value of 2. Furthermore, those words filtered out are tagged with a custom token that best identifies it. For example, if 'Mohammad' appeared once in the data, it will be filtered and tagged with \<unk-upper> as it starts with an upper case. Similarly, for the other cases.
# 
# Note: I tried applying lowercase on the words but obtained lower accuracy.
# 
# Next, I counted the transition and emission occurences and the tag occurences. From that I was able to calculate the transition and emission probabilities. I tried Laplace smoothing but I dropped that later on because I didn't manage to get the formula right (the sum of probabilities was not 1); it was t(s'|s) = (count(s->s')+1) / (count(s) + count (s,s') pairs).
# 
# Next, I implemented the Greedy and viterbi algorithms and used that on the dev dataset to get the accuracy and then on the test dataset to output the requested files. For both of these datasets, if a word is not in the vocabulary, I replaced it with a custom token that best identifies it (as explained earlier). This collectively helped boost the accuracy by roughly 1%.

# In[1]:


import re
import json
from collections import defaultdict


# # Task 1: Vocabulary Creation
# In this task, we will create a vocabulary using the training data. We will replace all words with occurences less than a threshold (default: 3) to a special token \<unk>. Finally, we will export this vocabulary to a file named vocab.txt where each line is in the format of 'word\tindex\toccurences', where index starts at 0 and occurences ordered in descending order. Moreover, the first line should be the \<unk> token with index 0.

# ## Extract words from the training dataset
# Read the dataset line by line and extract all words into a list.

# In[2]:


# Read the entire given training dataset
with open('data/train', 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

# Extract only the words from each line
words = []
for line in lines:
    if line:
        word = line.split('\t')[1] # format of line is [index, word, tag]
        words.append(word)


# ## Count the frequency of each word
# Use the words list extracted from previous step to count the frequency of each word.

# In[3]:


# Count each word
words_freq = defaultdict(int)
for word in words:
    words_freq[word] += 1


# ## Implement a way to handle unknown words
# Unknown words are words not present in the vocabulary which is bound to happen. Thus, the better we handle those cases the better the accuracy of the model becomes. Hence. I included a few cases to improve accuracy:
# <ol>
#     <li><b>unk-num</b>: when a word is a number or a fraction</li>
#     <li><b>unk-punc</b>: when a word contains punctuation</li>
#     <li><b>unk-all-caps</b>: when a word is in ALL CAPS</li>
#     <li><b>unk-upper</b>: when a word contains an upper case</li>
#     <li><b>unk-noun</b>: when a word contains a noun suffix</li>
#     <li><b>unk-verb</b>: when a word contains a verb suffix</li>
#     <li><b>unk-adj</b>: when a word contains an adjective suffix</li>
#     <li><b>unk-adv</b>: when a word contains an adverb suffix</li>
#     <li><b>unk</b>: default fallback for an unknown word</li>
# </ol>

# In[4]:


import string

# Returns true if the string is a fraction. Note, this is modified to the form 'a\/b' where a,b are integers
# Credit: https://stackoverflow.com/questions/38523110/regular-expression-is-fraction-python-regex
def is_fraction(string):
    return bool(re.search(r'^-?[0-9]+\\/0*[1-9][0-9]*$', string))

# Invoked when the word is not present in the vocab, we can analyze the word for hints
def get_unk_token(word):
    punctuation = set(string.punctuation)
    
    # Suffixes. Credit to Coursera NLP course Week 2 for this snippet / idea.
    # aswell as from: https://cl.lingfil.uu.se/~nivre/statmet/haulrich.pdf
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]
    
    # Check if the word is a number, or is a fraction (a\/b)
    if word.isnumeric() or is_fraction(word):
        return '<unk-num>'
    
    elif any(c in punctuation for c in word):
        return '<unk-punc>'
    
    # Check if the word is all in caps
    elif all(c.isupper() for c in word):
        return '<unk-all-caps>'
    
    # Check if the word has an uppercase
    elif any(c.isupper() for c in word):
        return '<unk-upper>'
    
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return '<unk-noun>'
    
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return '<unk-verb>'
    
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return '<unk-adj>'
    
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return '<unk-adv>'
    
    # Fallback
    return '<unk>'


# ## Build a vocabulary and filter words appearing less than the threshold
# By default, I used a threshold of 2. That means any word appearing once will not be part of the vocabulary. Moreover, the vocabulary will be sorted by descending order of frequency of the word, and the first item in the vocabulary is the special token \<unk>.

# In[5]:


# hyper-parameter for word cutoff based on occurences
threshold = 2
    
# Filter the words that appear less than the threshold
vocab = {}

# used with our custom tokens
custom_vocab = {
    '<unk>': 0,
    '<unk-num>': 0,
    '<unk-all-caps>': 0,
    '<unk-upper>': 0,
    '<unk-punc>': 0,
    '<unk-noun>': 0,
    '<unk-verb>': 0,
    '<unk-adj>': 0,
    '<unk-adv>': 0,
    '<start>': 0
}

for word in words:
    freq = words_freq[word]
    # The word doesnt appear a lot, so we filter it out
    if freq < threshold:
        # We check what kind of unknown word will it be
        unk_token = get_unk_token(word)
        custom_vocab[unk_token] += freq
    # The word surpassed the threshold, so we add it to the vocab
    else:
        vocab[word] = freq

# Sort the vocabulary by descending order of frequency of each word
vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

# In order to have <unk> at the first element of vocab, we'll use the insertion order guarantee of Python's 3.7+ dict
custom_vocab.update(vocab)
vocab = custom_vocab # add the vocab after the custom tokens


# ## Export vocabulary to file
# Write the vocabulary to disk, writing each word on its own line in the format <code>word index occurences</code> where the spaces are tabs.

# In[6]:


with open('vocab.txt', 'w') as f:
    # write each word in the format 'word\tindex\toccurences'
    for index, (word, frequency) in enumerate(vocab.items()):
        f.write(f'{word}\t{index}\t{frequency}')
        # Add new line if not on the last entry
        if index < len(vocab)-1:
            f.write('\n')


# ## Display statistics about the vocabulary

# In[7]:


print(f'Threshold: {threshold}')
print(f'Total size of vocabulary: {len(vocab)}')
print(f'Occurences of <unk>: {vocab["<unk>"]}')
print(f'Occurences of <unk-num>: {vocab["<unk-num>"]}')
print(f'Occurences of <unk-all-caps>: {vocab["<unk-all-caps>"]}')
print(f'Occurences of <unk-upper>: {vocab["<unk-upper>"]}')
print(f'Occurences of <unk-punc>: {vocab["<unk-punc>"]}')
print(f'Occurences of <unk-noun>: {vocab["<unk-noun>"]}')
print(f'Occurences of <unk-verb>: {vocab["<unk-verb>"]}')
print(f'Occurences of <unk-adj>: {vocab["<unk-adj>"]}')
print(f'Occurences of <unk-adv>: {vocab["<unk-adv>"]}')


# # Task 2: Model Learning
# In this task we will create an HMM from the training data. 
# 
# Note that when we parse the sentences from the training data, I inject a special token, <code>\<start></code>, at the start of each sentence respectively. This trick aids us in computing the probabilities for the hmm, specifically <code>t(s1)</code> becomes <code>t(s1|\<start>)</code>.

# ## Retrieve word and tag from line
# A helper function to parse a line into the word and tag. If the word is not in the vocabulary, it will use the unk categories defined above to best classify it, i.e. unseen numbers will get \<num>.

# In[8]:


# Given a line in the dataset, this will parse the tag and word
# If the word is not in the vocab, it will try to categorize it using the categories listed above
# Also, the special token <start> is used to indicate the start of a sentence
def get_word_tag(line, vocab):
    if not line.split():
        return '<start>', '<start>'
    else:
        index, word, tag = line.strip().split('\t')
        # Check if the word is in the vocab, and if its not, then assign a special token that aids in classifying it
        if word not in vocab:
            word = get_unk_token(word)
        return word, tag
    
# Given a line in the dataset, this will parse the line and return the word as is
# Also, the special token <start> is used to indicate the start of a sentence
def get_word_from_line(line):
    if not line.split():
        return '<start>'
    else:
        split = line.strip().split('\t')
        # test dataset: split = [index, word]
        # train/dev dataset: split = [index, word, tag]
        return split[1]


# ## Calculate transition and emission probabilities
# The closed-form for the emission and transition parameters in HMM are <code>t(s*|s) = count(s->s')/count(s)</code> and <code>e(x|s) = count(s->x)/count(s)</code> respectively, where <code>s</code> is current tag, <code>s'</code> is next tag, and <code>x</code> is the current word.
# 
# After calculating these probabilities, we export them as two dictionaries to <code>'hmm.json'</code>. The first dictionary, transition, contains items with pairs of <code>(s,s')</code> as key and <code>t(s'|s)</code> as value. The second dictionary, emission, contains items with pairs of <code>(s,x)</code> as key and <code>e(x|s)</code> as value.

# In[9]:


# Used to hold the previous state (tag) so it can be used for the key of transition probability
prev_tag = '<start>'

# Initialize HMM parameters
transitions = defaultdict(float)
emissions = defaultdict(float)
tag_freq = defaultdict(int) # used to compute count(s) denominator

for line in lines:
    # recall: empty lines will have <start>, <start> word, tag
    word, tag = get_word_tag(line, vocab)
    
    # transitions: key is (s, s') and value is count(s -> s') / count(s)
    transitions[(prev_tag, tag)] += 1
    
    # emissions: key is (s, x) and value is count(s -> x) / count(s)
    emissions[(tag, word)] += 1
    
    # counts the number of times a tag occured
    # used later to normalize transition and emission probabilities using count(s) denominator
    tag_freq[tag] += 1
    
    # update pointer so the transitions can use the correct key
    prev_tag = tag

# Extracts the possible tags
tags = tag_freq.keys()
    
# normalize transitions using the tag_freq (basically divide by count(s))
for key in transitions.keys():
    prev_tag, tag = key
    transitions[key] /= tag_freq[prev_tag]
    
# normalize emissions using the tag_freq (basically divide by count(s))
for key in emissions.keys():
    tag, word = key
    emissions[key] /= tag_freq[tag]
    
# Exports the transition and emission probabilities to the specified filename as json format
with open('hmm.json', 'w') as f:
    # Convert the tuple key to string key to be able to export as json
    t = {str(k): v for k, v in transitions.items()}
    e = {str(k): v for k, v in emissions.items()}
    json.dump({'transition': t, 'emission': e}, f)


# ## Display statistics regarding the HMM model

# In[10]:


transitions_params = len(transitions)
emissions_params = len(emissions)

print(f'Transition Parameters in HMM: {transitions_params}')
print(f'Emission Parameters in HMM: {emissions_params}')


# # Task 3: Greedy Decoding with HMM
# Here we implement and evalute the greedy decoding algorithm on the development data and report the accuracy. Then, we predict the pos tags on the test data and export the result into a file with format similar to the training data.

# ## Evalute Greedy HMM on dev dataset
# Load the dev dataset and predict the part-of-speech tags. Then, compare it with the expected tags and compute the accuracy.

# In[11]:


# Read the entire given development dataset
with open('data/dev', 'r') as f:
    dev_lines = f.readlines()
    dev_lines = [line.strip() for line in dev_lines]

# Count the number of tags predicted correctly
num_correct = 0
total = len(dev_lines)

# A simple way to keep a reference to the previous tag, used for t(s' | s)
prev_tag = '<start>'

for line in dev_lines:
    processed_word, expected_tag = get_word_tag(line, vocab)
    if processed_word in vocab:
        # Keep counter of which tag has the best likelihood
        best_tag = None
        best_tag_value = 0
        
        # for all possible tags, compute t(s'|s) * e(x|s), and choose the tag that gives the highest value
        for cur_tag in tags:
            # retrieve emission probability
            e = 0
            e_key = (cur_tag, processed_word)
            if e_key in emissions:
                e = emissions[e_key]
            # retrieve transition probability
            t = 0
            t_key = (prev_tag, cur_tag)
            if t_key in transitions:
                t = transitions[t_key]
            # keep the max value
            tag_value = t * e
            if best_tag is None or tag_value >= best_tag_value:
                best_tag = cur_tag
                best_tag_value = tag_value
        
        # count the number of correct predictions
        if best_tag == expected_tag:
            num_correct += 1
        
        prev_tag = best_tag

print(f'Greedy Decoding Accuracy: {num_correct/total:.3f}')


# ## Test Greedy HMM on test dataset
# Load the test dataset and predict the part-of-speech tags. Then, export the predictions to a <code>greedy.out</code> which has a format similar to that of the train dataset.

# In[12]:


# Read the entire given test dataset
with open('data/test', 'r') as f:
    test_lines = f.readlines()
    test_lines = [line.strip() for line in test_lines]

# A simple way to keep a reference to the previous tag, used for t(s' | s)
prev_tag = '<start>'

# A list to contain all of our part-of-speech predictions
predictions = []

for line in test_lines:
    word = get_word_from_line(line)
    
    processed_word = word
    if processed_word not in vocab:
        processed_word = get_unk_token(word)
        
    if processed_word in vocab:
        # Keep counter of which tag has the best likelihood
        best_tag = None
        best_tag_value = 0
        
        # for all possible tags, compute t(s'|s) * e(x|s), and choose the tag that gives the highest value
        for cur_tag in tags:
            # retrieve emission probability
            e = 0
            e_key = (cur_tag, processed_word)
            if e_key in emissions:
                e = emissions[e_key]
            # retrieve transition probability
            t = 0
            t_key = (prev_tag, cur_tag)
            if t_key in transitions:
                t = transitions[t_key]
            # keep the max value
            tag_value = t * e
            if best_tag is None or tag_value >= best_tag_value:
                best_tag = cur_tag
                best_tag_value = tag_value
        
        prev_tag = best_tag
        predictions.append((word, processed_word, best_tag))
        
# Export result
with open('greedy.out', 'w') as f:
    index = 1
    for prediction in predictions:
        word, processed_word, predicted_tag = prediction
        if processed_word == '<start>':
            f.write(f'\n')
            index = 1
        else:
            f.write(f'{index}\t{word}\t{predicted_tag}\n')
            index += 1


# # Task 4: Viterbi Decoding with HMM
# Here we implement and evalute the viterbi decoding algorithm on the development data and report the accuracy. Then, we predict the pos tags on the test data and export the result into a file with format similar to the training data.

# ## Evaluate Viterbi HMM on dev dataset
# Load the dev dataset and predict the part-of-speech tags. Then, compare it with the expected tags and compute the accuracy.

# In[13]:


# Backtrack to find the predicted tags using the viterbi and backpointer matrices
def extract_sequence(viterbi, backpointer, sentence_length):
    m = sentence_length
    prediction = []
    
    cur_tag = None
    cur_tag_value = 0
    for tag in tags:
        tag_value = viterbi[(m, tag)]
        if cur_tag is None or tag_value >= cur_tag_value:
            cur_tag = tag
            cur_tag_value = tag_value

    prediction.append(cur_tag)
    j = cur_tag
    
    while m > 1:
        i = backpointer[(m, j)]
        prediction.append(i)
        j = i
        m -= 1
    prediction.reverse()
    
    return prediction


# In[14]:


# Count the number of tags predicted correctly
num_correct = 0
total = len(dev_lines)

# Note the index in viterbi/backpointer are 1-based
index = 1

# Incrementally build the sentence to measure accuracy
sentence = []

# The viterbi matrix
viterbi = defaultdict(float) # key: (index, tag) and value: likelihood
backpointer = {} # key: (index, tag) and value: previous tag

for line in dev_lines:
    processed_word, expected_tag = get_word_tag(line, vocab)
    if processed_word in vocab:
        # start of sentence marker
        if processed_word == '<start>':
            index = 1
            
            # flush out the last sentence and reset vars for the new sentence
            if len(sentence) > 0:
                sequence = extract_sequence(viterbi, backpointer, len(sentence))
                for i in range(len(sequence)):
                    if sequence[i] == sentence[i][1]:
                        num_correct += 1
            
            # Reset the sentence and viterbi/backpointer matrix for the new sentece
            sentence = []
            viterbi = defaultdict(float) # key: (index, tag) and value: likelihood
            backpointer = {} # key: (index, tag) and value: previous tag
            continue
            
        # for the first word, initialize viterbi/backpointer matrices
        if index == 1:
            # for all possible tags, initialize it with the <start> tag
            for tag in tags:
                # retrieve emission probability
                e = 0
                e_key = (tag, processed_word)
                if e_key in emissions:
                    e = emissions[e_key]
                
                # retrieve transition probability
                t = 0
                t_key = ('<start>', tag)
                if t_key in transitions:
                    t = transitions[t_key]
                    
                # compute the final value
                final_value = t * e
                
                # update the matrices
                viterbi[(index, tag)] = final_value
                backpointer[(index, tag)] = None
        # for all other words, compute the recurrence relation
        else:
            # For all possible tags, select the tag that maximizes the viterbi entry
            for cur_tag in tags:
                viterbi[(index, cur_tag)] = -1
                # For all possible tags in the previous iteration
                for prev_tag in tags:
                    # retrieve previous value of viterbi entry
                    prev_viterbi_value = viterbi[(index-1, prev_tag)]

                    # retrieve emission probability
                    e = 0
                    e_key = (cur_tag, processed_word)
                    if e_key in emissions:
                        e = emissions[e_key]

                    # retrieve transition probability
                    t = 0
                    t_key = (prev_tag, cur_tag)
                    if t_key in transitions:
                        t = transitions[t_key]

                    # compute the final value
                    final_value = prev_viterbi_value * t * e

                    # If its a better likelihood, update its viterbi value as well as point to the tag for backtracking
                    if final_value > viterbi[(index, cur_tag)]:
                        viterbi[(index, cur_tag)] = final_value
                        backpointer[(index, cur_tag)] = prev_tag
        
        sentence.append((processed_word, expected_tag))
        index += 1

print(f'Viterbi Decoding Accuracy: {num_correct/total:.3f}')


# ## Test Viterbi HMM on test dataset
# Load the test dataset and predict the part-of-speech tags. Then, export the predictions to a <code>viterbi.out</code> which has a format similar to that of the train dataset.

# In[15]:


# Note the index in viterbi/backpointer are 1-based
index = 1

# Incrementally build the sentence to measure accuracy
sentence = []

# The viterbi matrix
viterbi = defaultdict(float) # key: (index, tag) and value: likelihood
backpointer = {} # key: (index, tag) and value: previous tag

# A list to contain all of our part-of-speech predictions
predictions = []

for line in test_lines:
    word = get_word_from_line(line)
    
    processed_word = word
    if processed_word not in vocab:
        processed_word = get_unk_token(word)
    
    if processed_word in vocab:
        # start of sentence marker
        if processed_word == '<start>':
            index = 1

            # flush out the last sentence and reset vars for the new sentence
            if len(sentence) > 0:
                sequence = extract_sequence(viterbi, backpointer, len(sentence))
                for i in range(len(sequence)):
                    # add to predictions: word, processed word, predicted tag
                    predictions.append((sentence[i][0], sentence[i][1], sequence[i]))
                predictions.append(('<start>', '<start>', '<start>'))
            
            # Reset the sentence and viterbi/backpointer matrix for the new sentece
            sentence = []
            viterbi = defaultdict(float) # key: (index, tag) and value: likelihood
            backpointer = {} # key: (index, tag) and value: previous tag
            continue
            
        # for the first word, initialize viterbi/backpointer matrices
        if index == 1:
            # for all possible tags, initialize it with the <start> tag
            for tag in tags:
                # retrieve emission probability
                e = 0
                e_key = (tag, processed_word)
                if e_key in emissions:
                    e = emissions[e_key]
                
                # retrieve transition probability
                t = 0
                t_key = ('<start>', tag)
                if t_key in transitions:
                    t = transitions[t_key]
                    
                # compute the final value
                final_value = t * e
                
                # update the matrices
                viterbi[(index, tag)] = final_value
                backpointer[(index, tag)] = None
        # for all other words, compute the recurrence relation
        else:
            # For all possible tags, select the tag that maximizes the viterbi entry
            for cur_tag in tags:
                viterbi[(index, cur_tag)] = -1
                # For all possible tags in the previous iteration
                for prev_tag in tags:
                    # retrieve previous value of viterbi entry
                    prev_viterbi_value = viterbi[(index-1, prev_tag)]

                    # retrieve emission probability
                    e = 0
                    e_key = (cur_tag, processed_word)
                    if e_key in emissions:
                        e = emissions[e_key]

                    # retrieve transition probability
                    t = 0
                    t_key = (prev_tag, cur_tag)
                    if t_key in transitions:
                        t = transitions[t_key]

                    # compute the final value
                    final_value = prev_viterbi_value * t * e

                    # If its a better likelihood, update its viterbi value as well as point to the tag for backtracking
                    if final_value > viterbi[(index, cur_tag)]:
                        viterbi[(index, cur_tag)] = final_value
                        backpointer[(index, cur_tag)] = prev_tag
        
        sentence.append((word, processed_word))
        index += 1
        
# Export result
with open('viterbi.out', 'w') as f:
    index = 1
    for i, (word, processed_word, predicted_tag) in enumerate(predictions):
        if processed_word == '<start>':
            f.write(f'')
            index = 1
        else:
            f.write(f'{index}\t{word}\t{predicted_tag}')
            index += 1
        # only add a trailing newline if we are not at the last prediction
        if i < len(predictions)-1:
            f.write('\n')

