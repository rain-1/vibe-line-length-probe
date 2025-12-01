# Model

use huggingface transformers and pytorch

we will use https://huggingface.co/google/gemma-3-270m
get this model up and running and test it works

# Data set generation

grab a data set https://huggingface.co/suayptalha/Poetry-Foundation-Poems
we want the Poem column
tokenize it

to compute our 'target' we need a "current line length" dataset
this will be easy to compute, show me the results (tokenized string from a poem alongside the current line length data) so I can confirm it's right
