# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A2: Vector Semantics
#
# **By Ricardo Muñoz Sánchez, Nikolai Ilinykh, Mehdi Ghanimifard, Wafia Adouane and Simon Dobnik.**
#
# The lab is an exploration and learning exercise to be done in a group and also in discussion with the teachers and other students.
#
# Before starting, please read the instructions on how to work in groups on Canvas.
#
# Write all your answers and the code in the appropriate boxes below.
#
# In this lab we will look at how to build distributional semantic models from corpora and use semantic similarity captured by these models to do semantic tasks. We are also going to examine how different vector composition functions for vectors work in approximating semantic similarity of phrases when compared to human judgements.
#
# **Dependencies:** This lab uses the code from `dist_erk.py` to find word counts and build vectors. For details how this is done, go through the the code and the comments provided there. `dist_erk.py` uses Spacy and NLTK, and here we use SciPY and scikit-learn so make sure they are installed. Instructions how to do this:
#
#   - https://scipy.org/beginner-install/
#   - https://scikit-learn.org/stable/install.html
#   - https://spacy.io/usage
#   - https://www.nltk.org/install.html
#
# **On using generative AI for this assignment:** For this lab, the use of generative AI is permitted as a supporting tool, provided it is done in a responsible and conscious manner and that you state clearly with each question how it was used. However, generative AI must never replace the intellectual work you are expected to carry out. Note that the purpose of this lab is to learn some basic coding of the main neural architectures used in natural language processing. You are responsible for ensuring that such tools are used in a way that supports the development of the skills the module is designed to promote. It is your responsibility to ensure that submitted work is the result of independent intellectual effort.
#
# **Getting help:** We encourage you to use Canvas discussions to post questions and interact with teachers and also each other. Provide youseful tips, but of course do not reveal the exact answer across the groups as each group should should work out their own solutions. Remember that in most cases there is also not a single correct answer.

# %%
# %pip install -r packages.txt

# %%
# We also need models and datasets for Spacy

import spacy

spacy.cli.download('en_core_web_sm')

# You only need to run this cell once.
# You *have to* restart the kernel after downloading the model!

# %%
# the following command imports all the methods from the dist_erk file
from dist_erk import *

# %% [markdown]
# ## 1. Loading a corpus
#
# To train a distributional model, we first need a sufficiently large collection of texts which contain different words used frequently enough in different contexts. Here we will use a section of the Wikipedia corpus `wikipedia.txt` stored in `wikipedia.zip`.
#
# When unpacked, the file is 151mb, hence if you are using the MLT servers you should store it in a temporary folder outside your home and adjust the `corpus_dir` path below. It may already exist in `/srv/data/computational-semantics/`.

# %%
corpus_dir = './wikipedia'

# %% [markdown]
# ## 2. Building a model
#
# Now you are ready to build the model.  
# Using the methods from the code imported above build three word matrices with 1000 dimensions as follows:  
#
# (i) with raw counts (saved to a variable `space_1k`);  
# (ii) with PPMI (`ppmispace_1k`);  
# (iii) with reduced dimensions SVD (`svdspace_1k`).  
# For the latter use `svddim=5`. **[5 marks]**
#
# Your task is to replace `...` with function calls to functions from `dist_erk.py`.
#
# Do not despair if the code takes a bit long to run!
# It took me about 9 minutes for the cell below.

# %%
numdims = 1000
svddim = 5

# Which words to use as targets and context words?
# We need to count the words and keep only the N most frequent ones
# Which function would you use here with which variable?
ktw = do_word_count(corpus_dir, numdims)

wi = make_word_index(ktw)  # word index
words_in_order = sorted(wi.keys(), key=lambda w: wi[w])  # sorted words

# Create different spaces (the original matrix space, the ppmi space, the svd space)
# Which functions with which arguments would you use here?
print('create count matrices')
space_1k = make_space(corpus_dir, wi, numdims)
print('ppmi transform')
ppmispace_1k = ppmi_transform(space_1k, wi)
print('svd transform')
svdspace_1k = svd_transform(ppmispace_1k, numdims, svddim)
print('done.')

# %%
# now, to test the space, you can print vector representation for some words
print('house:', space_1k['house'])

# %% [markdown]
# Oxford Advanced Dictionary has 185,000 words, hence 1,000 words is not representative. We trained a model with 10,000 words, and 50 dimensions on truncated SVD. All matrices are available in the folder `pretrained` of the `wikipedia.zip`file. These are `ktw_wikipediaktw.npy`, `raw_wikipediaktw.npy`, `ppmi_wikipediaktw.npy`, `svd50_wikipedia10k.npy`. Make sure they are in your path as we load them below.

# %%
import numpy as np

numdims = 10000
svddim = 50

print('Please wait...')
ktw_10k = np.load('./wikipedia/pretrained/ktw_wikipediaktw.npy', allow_pickle=True)
space_10k = np.load('./wikipedia/pretrained/raw_wikipediaktw.npy', allow_pickle=True).tolist()
ppmispace_10k = np.load('./wikipedia/pretrained/ppmi_wikipediaktw.npy', allow_pickle=True).tolist()
svdspace_10k = np.load('./wikipedia/pretrained/svd50_wikipedia10k.npy', allow_pickle=True).tolist()
print('Done.')


# %%
# testing semantic space
print('house:', space_10k['house'])

# %% [markdown]
# ## 3. Testing semantic similarity
#
# The file `similarity_judgements.txt` contains 7,576 pairs of words and their lexical and visual similarities (based on the pictures) collected through crowd-sourcing using Mechanical Turk as described in [1]. The scores range from 1 (highly dissimilar) to 5 (highly similar). You can find more details about how they were collected in the papers.
#
# The following code will transform similarity scores into a Python-friendly format:

# %%
word_pairs = []  # test suit word pairs
semantic_similarity = []
visual_similarity = []
test_vocab = set()

for index, line in enumerate(open('similarity_judgements.txt')):
    data = line.strip().split('\t')
    if index > 0 and len(data) == 3:
        w1, w2 = tuple(data[0].split('#'))
        # Checks if both words from each pair exist in the word matrix.
        if w1 in ktw_10k and w2 in ktw_10k:
            word_pairs.append((w1, w2))
            test_vocab.update([w1, w2])
            semantic_similarity.append(float(data[1]))
            visual_similarity.append(float(data[2]))

print('number of available words to test:', len(test_vocab - (test_vocab - set(ktw))))
print('number of available word pairs to test:', len(word_pairs))
#list(zip(word_pairs, visual_similarity, semantic_similarity))

# %% [markdown]
# We are going to test how the cosine similarity between vectors of each of the three spaces (normal space, ppmi, svd) compares with the human similarity judgements for the words in the similarity dataset. Which of the three spaces best approximates human judgements?
#
# For comparison of several scores, we can use [the Spearman correlation coefficient](https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient) which is implemented in `scipy.stats.spearmanr` [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html). The values of the Sperman correlation coefficient range from -1, 0 to 1, where 0 indicates no correlation, 1 perfect correaltion and -1 negative correlation. Hence, the greater the number the better the similarity scores align. The p values tells us if the coefficient is statistically significant. For this to be the case, it must be less than or equal to $< 0.05$.
#
# Here is how you can calculate the Spearman correlation coefficient betweeen the scores of visual similarity and semantic similarity of the available words in the test suite:

# %%
from scipy import stats

rho, pval = stats.spearmanr(semantic_similarity, visual_similarity)
print("""Visual Similarity vs. Semantic Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))


# %% [markdown]
# Let's now calculate the cosine similarity scores of all word pairs in an ordered list using all three matrices. Some calculations will give you errors, why? **[6 marks]**

# %%
raw_similarities = [cosine(w1, w2, space_10k) for w1, w2 in word_pairs]
ppmi_similarities = [cosine(w1, w2, ppmispace_10k) for w1, w2 in word_pairs]
svd_similarities = [cosine(w1, w2, svdspace_10k) for w1, w2 in word_pairs]

# %% [markdown]
# Calculate correlation coefficients between lists of similarity scores and the real semantic similarity scores from the experiment. The scores of what model best correlates them? Is this expected? **[6 marks]**

# %%
rho, pval = stats.spearmanr(semantic_similarity, raw_similarities)
print("""Raw Similarity vs. Semantic Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))  # your code should go here

# %%
rho, pval = stats.spearmanr(semantic_similarity, ppmi_similarities)
print("""PPMI Similarity vs. Semantic Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))  # your code should go here

# %%
rho, pval = stats.spearmanr(semantic_similarity, svd_similarities)
print("""SVD Similarity vs. Semantic Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))  # your code should go here

# %% [markdown]
# **Your answer should go here:**

# %% [markdown]
# The PPMI model correlates best with the human scores. This is expected because raw counts are skewed by common but uninformative words like "the" or "and." PPMI filters out this noise and focuses on meaningful co-occurrences. SVD performs slightly worse than PPMI in this case, likely because the compression of dimensions leads to some loss of specific data. Therefore, PPMI provides the most accurate representation of similarity here.

# %% [markdown]
# We can also calculate correlation coefficients between lists of cosine similarity scores and the real visual similarity scores from the experiment. Which similarity model best correlates with them? How do the correlation coefficients compare with those from the previous comparison - and can you speculate why do we get such results? **[7 marks]**

# %%
rho, pval = stats.spearmanr(visual_similarity, raw_similarities)
print("""Raw Similarity vs. Visual Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))  # your code should go here# Your code should go here...

# %%
rho, pval = stats.spearmanr(visual_similarity, ppmi_similarities)
print("""PPMI Similarity vs. Visual Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))  # your code should go here

# %%
rho, pval = stats.spearmanr(visual_similarity, svd_similarities)
print("""SVD Similarity vs. Visual Similarity:
rho     = {:.4f}
p-value = {:.4f}""".format(rho, pval))  # your code should go here

# %% [markdown]
# **Your answer should go here:**

# %% [markdown]
# The PPMI model correlates best with the visual scores. However, compared to the results from the previous comparison, these scores are lower across the board.
#
# Our speculation is that because the models are built using text-only data, they are naturally better at capturing conceptual meanings than visual ones. The correlation scores for visuals are still relatively high likely because visually similar things often appear in similar contexts (for example, "deer" and "elk" both appearing in stories about forests), allowing the model to learn about visual features as a side effect of reading the text.

# %% [markdown]
# ## 4. Operations on similarities

# %% [markdown]
# We can perform mathematical operations on vectors to derive meaning predictions.
#
# For example, we can perform `king - man` and add the resulting vector to `woman` and we hope to get the vector for `queen`. What would be the result of `stockholm - sweden + denmark`? Why? **[3 marks]**
#
# If you want to learn more about vector differences between words (and words in analogy relations), check this paper [4].

# %% [markdown]
# **Your answer should go here:**

# %% [markdown]
# The answer should be `copenhagen`, that's because the vector difference  `stockholm - sweden` indicates the relationship between the country and the capital city. When we add `denmark`, it points towards the capital of Denmark, which should be `copenhagen`.

# %% [markdown]
# Here is some code that allows us to calculate such comparisons.

# %%
from scipy.spatial import distance


def normalize(vec):
    return vec / veclen(vec)


def find_similar_to(vec1, space):
    # vector similarity funciton
    #sim_fn = lambda a, b: 1-distance.euclidean(normalize(a), normalize(b))
    #sim_fn = lambda a, b: 1-distance.correlation(a, b)
    #sim_fn = lambda a, b: 1-distance.cityblock(normalize(a), normalize(b))
    #sim_fn = lambda a, b: 1-distance.chebyshev(normalize(a), normalize(b))
    #sim_fn = lambda a, b: np.dot(normalize(a), normalize(b))
    sim_fn = lambda a, b: 1 - distance.cosine(a, b)

    sims = [
        (word2, sim_fn(vec1, space[word2]))
        for word2 in space.keys()
    ]
    return sorted(sims, key=lambda p: p[1], reverse=True)


# %% [markdown]
# Here is how you apply this code. Comment on the results you get. **[3 marks]**

# %%
short = normalize(svdspace_10k['short'])
light = normalize(svdspace_10k['light'])
long = normalize(svdspace_10k['long'])
heavy = normalize(svdspace_10k['heavy'])

find_similar_to(light - (heavy - long), svdspace_10k)[:10]

# %% [markdown]
# **Your answer should go here:**

# %% [markdown]
# The output is not what we thought before. The first result is `long` instead of `short`. We guess this may be because `light` and `heavy` are considered as opposite ends of the same weight dimension, so they partly cancel each other out, and the result stays closer to `long`. Some other words, such as `wide` and `length`, are still understandable, because they are also related to physical dimension. Words like `above`、 `around` and `circle` may also be loosely related to spatial dimension, so they are not completely random. However, we were confused by outputs such as `sun`、`just`、`each` and `almost`. For `sun`, we think maybe the meaning of `light` here is not only understood as the opposite of `heavy`, but may also be interpreted as sunlight or brightness. That made us realize that ambiguity in word meaning is a big problem when using vectors, because one vector may mix different senses of the same word. For the other unrelated outputs, they may reflect noise in the vector space or limitations of this analogy operation. Overall, the result is only partly convincing, because a few words are semantically relevant, but the model does not return a clear expected opposite such as `short`.

# %% [markdown]
# Find 5 similar pairs of pairs of words and test them. Hint: google for `word analogies examples`. You can also construct analogies that are not only lexical but also express other relations such as grammatical relations, e.g. `see, saw, leave, ?` or analogies that are based on world knowledge as in `question-words.txt` from the [Google analogy dataset](http://download.tensorflow.org/data/questions-words.txt) described in [3]. Does the resulting vector similarity confirm your expectations? Remember you can only do this test if the words are contained in our vector space with 10,000 dimensions. **[10 marks]**

# %%
# Your code should go here...
see = normalize(svdspace_10k['see'])
saw = normalize(svdspace_10k['saw'])
leave = normalize(svdspace_10k['leave'])
print("see - saw + leave:")
print(find_similar_to(see - saw + leave, svdspace_10k)[:5])

china = normalize(svdspace_10k['china'])
chinese = normalize(svdspace_10k['chinese'])
sweden = normalize(svdspace_10k['sweden'])
print("\nchina - chinese + sweden:")
print(find_similar_to(china - chinese + sweden, svdspace_10k)[:5])

girl = normalize(svdspace_10k['girl'])
boy = normalize(svdspace_10k['boy'])
woman = normalize(svdspace_10k['woman'])
print("\ngirl - boy + woman:")
print(find_similar_to(girl - boy + woman, svdspace_10k)[:5])

beijing = normalize(svdspace_10k['beijing'])
china = normalize(svdspace_10k['china'])
paris = normalize(svdspace_10k['paris'])
print("\nbeijing - china + paris:")
print(find_similar_to(beijing - china + paris, svdspace_10k)[:5])

large = normalize(svdspace_10k['large'])
largest = normalize(svdspace_10k['largest'])
small = normalize(svdspace_10k['small'])
print("\nlarge - largest + small:")
print(find_similar_to(large - largest + small, svdspace_10k)[:5])

# %% [markdown]
# **see - saw + leave** → expected: `left`
# It doesn't work. `leave` itself appears first instead of `left`. This is probably because irregular verbs like `see` and `saw` are tricky for the model, since their past tense forms don't follow a consistent pattern.
#
# **china - chinese + sweden** → expected: `swedish`
# It doesn't work, the model returned European countries instead of `swedish`. This is probably because `chinese` has multiple meanings (language, nationality, ethnicity), so the model gets confused.
#
# **girl - boy + woman** → expected: `man`
# It works, `man` appears in third place, only behind the input words themselves.This suggests that gender relations are well captured in the vector space, which is expected.
#
# **beijing - china + paris** → expected: `france`
# It doesn't work, the model returned several European cities instead of a country, which suggests that it captured the relation of “city” rather than “capital of.”
#
# **large - largest + small** → expected: `smallest`
# It doesn't work, `smallest` does not appear in the top five, but `smaller` shows up in fourth place, so the direction is partly correct. The problem is that `large`, `largest`, and `small` are already very close in the vector space, which makes the difference weaker.

# %% [markdown]
# ## 5. Semantic composition and phrase similarity **[20 marks]**
#
# In this task, we are going to examine how the composed vectors of phrases by different semantic composition functions/models introduced in [2] correlate with human judgements of similarity between phrases. We will use the dataset from this paper which is stored in `mitchell_lapata_acl08.txt`. If you are interested about further details about this task also refer to this paper.
#
# (i) Process the dataset. The dataset contains human judgemements of similarity between phrases recorded one per line. The first column indicates the id of a participant making a judgement (`participant`), the next column is `verb`, followed by `noun` and `landmark`. From these three columns we can construct phrases that were compared by human informants, namely `verb noun` vs `verb landmark`. The next column `input` indicates a similarity score a participant assigned to a pair of such phrases on a scale from 1 to 7 where 1 is lowest and 7 is highest. The last column `hilo` groups the phrases into two sets: phrases where we expect low and phrases where we expect high similarity scores. This is because we want to test our compositional functions on two tasks and examine whether a function is discriminative between them. Correlation between scores could also be due to other reasons than semantic similarity and hence good prediction on both tasks simultaneously shows that a function is truly discriminating the phrases using some semantic criteria.
#
# For extracting information you can use the code from above to start with. How to structure this data is up to you - a dictionary-like format would be a good choice. Remember that each example was judged by several participants and phrases will repeat in the dataset. Therefore, you have to collect all judgments for a particular set of phrases and average them. This will become useful in step (iii).
#
# (ii) Compose the vectors of the extracted word pairs by testing different compositional functions, for example simple additive, simple multiplicative and combined models from [2]. Your task is to take a pair of phrases, e.g. the first example in the dataset `stray thought` and `stray roam` and for each phrase compute a composition of the vectors of their words using these functions, using one function per experiment run. For each phrase you will get a single vector. You can encode the words with any vector space introduced earlier (standard space, ppmi or svd) but your code should be structured in a way that it will be easy to switch between them. Finally, take the resulting (composed) vectors of phrase pairs in the dataset and calculate a cosine similarity between them.
#
# (iii) Now you have cosine similairity scores between vectors of phrases but how do they compare with the average human scores that you calculated from the individual judgements from the `input` column of the dataset for the same phrases? Calculate Spearman rank correlation coefficient between two lists of the scores both for the `high` and the `low` task . 
#
# We use the Spearmank rank correlation coefficient (or Spearman's rho) rather than Peason's correlation coefficent because we cannot compare cosine scores with human judgements directly. Cosine is a constinuous measure and human judgements are expressed as ranks. Also, we cannot say if 0.28 to 1 is the same (or different) to 6 to 7 in the human scores.  The Spearman rank correlation coeffcient turns the scores for all examples within each group first to ranks and then these ranks are correlated (or approximated to a linear function). 
#
# In the end you should get a table similar to the one below from the paper. What is the best compositional function from those that you evaluated with your vector spaces and why?
#
# <img src="res.png" alt="drawing" width="500"/>
#
# Note that you might not get results in the same range as those in the paper.
# That is ok, a good interpretation of results and discussion why sometimes they are not as good as you would expect is better than giving the best performing results with little to no analysis.
#

# %% [markdown]
# ⚠️ Experimenting with building higher dimension vectors.

# %%
numdims = 50000

ktw = do_word_count(corpus_dir, numdims)

wi = make_word_index(ktw)  # word index
words_in_order = sorted(wi.keys(), key=lambda w: wi[w])  # sorted words

print('create count matrices')
space_50k = make_space(corpus_dir, wi, numdims)
print('ppmi transform')
ppmispace_50k = ppmi_transform(space_50k, wi)

# %% [markdown]
# If you had already built the vectors, skip this.

# %%
import numpy as np

# save the target words list (useful for keeping track of the vocabulary)
np.save('./wikipedia/pretrained/ktw_wikipedia50k.npy', ktw)

# save the raw count space
np.save('./wikipedia/pretrained/raw_wikipedia50k.npy', np.array(space_50k))

# save the PPMI space
np.save('./wikipedia/pretrained/ppmi_wikipedia50k.npy', np.array(ppmispace_50k))

print("Successfully saved 50k PPMI space to disk.")

# %% [markdown]
# Load the vectors.

# %%
print('Please wait...')
ktw_20k = np.load('./wikipedia/pretrained/ktw_wikipedia20k.npy', allow_pickle=True)
space_20k = np.load('./wikipedia/pretrained/raw_wikipedia20k.npy', allow_pickle=True).tolist()
ppmispace_20k = np.load('./wikipedia/pretrained/ppmi_wikipedia20k.npy', allow_pickle=True).tolist()

# %%
# (i) - Process the data
# your code should go here

from collections import defaultdict

# Choose which semantic space to use (ppmispace_10k is usually the best performing)
# vector_space = ppmispace_10k
vector_space = ppmispace_20k
vector_space_name = 'PPMI 20k'

unique_dict = defaultdict(list)
skipped_count = 0
total_lines = 0

with open('mitchell_lapata_acl08.txt', 'r') as f:
    next(f)  # skip header

    for line in f:
        total_lines += 1
        data = line.strip().split()
        if len(data) < 6:
            continue

        # normalize to lowercase and extract values
        verb, noun, landmark = data[1].lower(), data[2].lower(), data[3].lower()
        user_input, hilo = data[4], data[5].lower()

        # check if all words are present in our vector space vocabulary
        if all(word in vector_space for word in [verb, noun, landmark]):
            key = (verb, noun, landmark, hilo)
            unique_dict[key].append(float(user_input))
        else:
            skipped_count += 1

# Calculate average human scores for each unique phrase key
phrase_score_dict = {k: sum(v) / len(v) for k, v in unique_dict.items()}

print(f"Using vector space: {vector_space_name}")
print(f"Total lines in file: {total_lines}")
print(f"Lines skipped (missing vocabulary): {skipped_count}")
print(f"Total unique phrase keys retained: {len(phrase_score_dict)}")

# %%
# (ii) - Compose the vectors of the extracted word pairs by testing different compositional functions
# your code should go here

import pandas as pd


def additive_composition(u, v):
    return u + v


def multiplicative_composition(u, v):
    return u * v


def combined_composition(verb, noun, verb_weight=0.3, noun_weight=0.7):
    return (verb * verb_weight) + (noun * noun_weight)


def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity using the same logic and veclen() function from dist_erk.py.
    """
    vlen1 = veclen(vec1)
    vlen2 = veclen(vec2)

    if vlen1 == 0.0 or vlen2 == 0.0:
        return 0.0
    else:
        # using the dot product logic from dist_erk.py
        dotproduct = np.sum(vec1 * vec2)
        return dotproduct / (vlen1 * vlen2)


results_data = []

for (verb, noun, landmark, hilo), human_avg in phrase_score_dict.items():
    # get word vectors
    v_vec = vector_space[verb]
    n_vec = vector_space[noun]
    l_vec = vector_space[landmark]

    # store everything in a dictionary for this row
    row = dict(verb=verb, noun=noun, landmark=landmark, hilo=hilo, human_score=human_avg)

    # calculate similarities for each model using the dist_erk logic
    row['noncomp_similarity'] = cosine_similarity(v_vec, l_vec)
    row['add_similarity'] = cosine_similarity(additive_composition(v_vec, n_vec),
                                              additive_composition(v_vec, l_vec))
    row['mul_similarity'] = cosine_similarity(multiplicative_composition(v_vec, n_vec),
                                              multiplicative_composition(v_vec, l_vec))
    row['com_similarity'] = cosine_similarity(combined_composition(v_vec, n_vec),
                                              combined_composition(v_vec, l_vec))

    results_data.append(row)

# save results as a DataFrame
results_df = pd.DataFrame(results_data)
print(f"Calculated {len(results_df)} similarity rows.")
results_df.head(n=10)

# %%
# (iii) - Compare the cosine similarity scores between vectors of phrases with the average human scores
# your code should go here

from scipy import stats

# define the models
# column names must match from results_df in Part (ii)
models = [
    ("NonComp", "noncomp_similarity"),
    ("Add", "add_similarity"),
    ("Multiply", "mul_similarity"),
    ("Combined", "com_similarity")
]

results = []

for label, col_name in models:
    # 1. calculate overall Spearman correlation with human judgments
    rho, pval = stats.spearmanr(results_df['human_score'], results_df[col_name])

    # 2. calculate mean similarity for High vs Low categories
    mean_high = results_df[results_df['hilo'] == 'high'][col_name].mean()
    mean_low = results_df[results_df['hilo'] == 'low'][col_name].mean()

    # 3. formatting stars for significance (p < 0.01 = **, p < 0.05 = *)
    stars = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")

    results.append({
        "Model": label,
        "High Mean": f"{mean_high:.3f}",
        "Low Mean": f"{mean_low:.3f}",
        "Spearman ρ": f"{rho:.3f}{stars}",
        "p-value": f"{pval:.4f}"
    })

df = pd.DataFrame(results)
print("Comparison of Compositional Models against Human Judgments:")
df

# %% [markdown]
# **Any comments/thoughts should go here:**
#
# We tried four models: non-compositional, additive, multiplicative, and combined across four different vector spaces (10k, 20k, 25k, 50k).
#
# What we found:
#
# Our best result was actually with the smallest space. Using the 10k PPMI space, the multiplicative model got rho = 0.714 (p = 0.046). The 10k space only contains the most frequent words, so the vectors for words like "bow", "boom" and "gun" are very dense and reliable since they appear thousands of times in the corpus.
#
# As we increased the vocabulary to 20k, 25k and 50k, more phrase pairs survived the vocabulary filter (from 8 up to 56) but the results actually got worse. We think this is because the newly included words are rarer, so their vectors are noisier and less reliable.
#
# By 50k, multiplicative rho had dropped to -0.140.The additive model was more stable across spaces but never really discriminated well between high and low similarity pairs, with high and low means staying very close together throughout. 
#
# We think the paper got much better results mainly because they built space with full vocabulary coverage of all dataset words, giving them all 120 phrase pairs to work with instead of our 8 to 56. With a proper coverage, we would expect our multiplicative results to become more stable and significant too.
#
# Overall, multiplicative composition seems to be the best function for this task, but it really depends on having good quality vectors. That feels like the main takeaway from our experiments.
#
# ## Results Documentation
#
# ### Pretrained PPMI 10k provided
#
# Using vector space: PPMI 10k\
# Total lines in file: 3600\
# Lines skipped (missing vocabulary): 3360\
# Total unique phrase keys retained: 8
#
# |  | Model | High Mean | Low Mean | Spearman ρ | p-value |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | 0 | NonComp | 0.053 | 0.053 | 0.000 | 1.0000 |
# | 1 | Add | 0.528 | 0.550 | -0.571 | 0.1390 |
# | 2 | Multiply | 0.132 | 0.033 | 0.714\* | 0.0465 |
# | 3 | Combined | 0.224 | 0.218 | 0.167 | 0.6932 |
#
# ### PPMI 20k
#
# Using vector space: PPMI 20k\
# Total lines in file: 3600\
# Lines skipped (missing vocabulary): 3060\
# Total unique phrase keys retained: 18
#
# |  | Model | High Mean | Low Mean | Spearman ρ | p-value |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | 0 | NonComp | 0.039 | 0.039 | -0.042 | 0.8698 |
# | 1 | Add | 0.481 | 0.490 | -0.116 | 0.6477 |
# | 2 | Multiply | 0.083 | 0.058 | 0.169 | 0.5018 |
# | 3 | Combined | 0.193 | 0.190 | -0.063 | 0.8039 |
#
# ### PPMI 50k
#
# Using vector space: PPMI 50k\
# Total lines in file: 3600\
# Lines skipped (missing vocabulary): 1920\
# Total unique phrase keys retained: 56
#
# |  | Model | High Mean | Low Mean | Spearman ρ | p-value |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | 0 | NonComp | 0.029 | 0.029 | 0.236 | 0.0797 |
# | 1 | Add | 0.364 | 0.348 | 0.162 | 0.2330 |
# | 2 | Multiply | 0.097 | 0.094 | -0.140 | 0.3044 |
# | 3 | Combined | 0.129 | 0.122 | 0.158 | 0.2442 |
#

# %% [markdown]
# # Literature
#
# [1] C. Silberer and M. Lapata. Learning grounded meaning representations with autoencoders. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics, pages 721–732, Baltimore, Maryland, USA, June 23–25 2014 2014. Association for Computational Linguistics.  
#
# [2] Mitchell, J., & Lapata, M. (2008). Vector-based Models of Semantic Composition. In Proceedings of ACL-08: HLT (pp. 236–244). Association for Computational Linguistics.
#   
# [3] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013.
#
# [4] E. Vylomova, L. Rimell, T. Cohn, and T. Baldwin. Take and took, gaggle and goose, book and read: Evaluating the utility of vector differences for lexical relation learning. arXiv, arXiv:1509.01692 [cs.CL], 2015.

# %% [markdown]
# ## Statement of contribution
#
# Briefly state how many times you have met for discussions, who was present, to what degree each member contributed to the discussion and the final answers you are submitting.

# %% [markdown]
# We first read through the assignment individually and had a short discussion before the first lab session. We then met in person during the lab sessions. Other than that, we mainly keep in touch via WhatsApp all the time.
#
# All the team members were active in participation.
#
# Yitong tackled part 4 and partially 5.\
# Mamitha tackled part 5.\
# Sana tackled part 2 and 3's code.\
# Eugene tackled part 2, 3 and 5.
#
# Everyone then reviewed each other's work and help to troubleshoot if any problem arises.
#

# %% [markdown]
# ## Marks
#
# The assignment is marked on a 7-level scale where 4 is sufficient to complete the assignment; 5 is good solid work; 6 is excellent work, covers most of the assignment; and 7: creative work. 
#
# This assignment has a total of 60 marks. These translate to grades as follows: 1 = 17% 2 = 34%, 3 = 50%, 4 = 67%, 5 = 75%, 6 = 84%, 7 = 92% where %s are interpreted as lower bounds to achieve that grade.
