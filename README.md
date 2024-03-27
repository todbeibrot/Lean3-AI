# Lean3-AI
Code for training an AI for Lean 3. The results are not useful.

It consists of multiple agents. One is for generating theorems while the others are trying to solve those. The reward for the theorem generation depends on the performance of the solvers. It is supposed to keep the tasks for the solvers manageable. In practice it doesn't work very well. Most likely for technical matters and the lack of computational power.

In order to use it, you need an instance of lean-gym (https://github.com/openai/lean-gym), replacing the files with the files in the folder "lean-gym". You also need a list of theorems and definitions to use.

If you are interested in me uploading training graphs, models, generated theorems, the word2vec model or lists of definitions/theorems to use, leave a message.
