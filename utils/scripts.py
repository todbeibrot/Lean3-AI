from os import listdir, remove
from os.path import join
import gensim
from typing import List


def merge_lemmas(dir_path: str, remove_files: bool = False):
    file_list = listdir(dir_path)
    # Collect all lemmas
    lemmas = []
    for file in file_list:
        file_path = join(dir_path, file)
        assert file.endswith('.lean') or file.endswith('.txt'), f'not a text or lean file: {file}'
        lemmas += open(file_path, 'r', encoding='utf-8').readlines()
        if remove_files:
            remove(file_path)
    # Write everything in one file
    lemmas = list(set(lemmas))
    for id, lemma in enumerate(lemmas):
        if 'rw aaa_unfold' in lemma:
            lemmas[id] = lemma[:lemma.find('rw aaa_unfold')] + 'any_goals {exact true}, clear_trivial\n'
    open(join(dir_path, 'all.lean'), 'w', encoding='utf-8').writelines(lemmas)

def merge_corpi(dir_path: str, remove_files: bool = False):
    corpus = []
    file_list = listdir(dir_path)
    for file in file_list:
        corpus += open(join(dir_path, file), 'r', encoding='utf-8').readlines()
        if remove_files:
            remove(join(dir_path, file))
    corpus = list(set(corpus))
    open(join(dir_path, 'corpus.txt'), 'w', encoding='utf-8').writelines(corpus)

def build_word2vec(corpus_path: str, model_save_path: str, workers=10, min_count=1, vector_size=10, epochs=100):
    model = gensim.models.Word2Vec(corpus_file=corpus_path, workers=workers, min_count=min_count, vector_size=vector_size, epochs=epochs)
    model.save(model_save_path)

def delete_proofs(path, new_path):
    file = open(path, 'r', encoding='utf-8').read()
    new_file = open(new_path, 'w', encoding='utf-8')
    begin = 0
    end = 0
    while begin >= 0:
        begin = file.find('begin', end)
        if begin != -1 and end != -1:
            new_file.write(file[end:begin] + 'sorry')
        end = file.find('end', begin) + 3
    
def get_theorems(path):
    theorems = []
    add_next_word = False
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        if 'theorem' in line:
            words = line.split()
            for word in words:
                if add_next_word:
                    theorems.append(word)
                if 'theorem' in word:
                    add_next_word = True
                else:
                    add_next_word = False
    return theorems

