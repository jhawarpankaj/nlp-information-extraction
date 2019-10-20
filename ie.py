import nltk
import re
import sys


# Fill in the pattern (see Part 2 instructions)
NP_grammar = 'NP: {<.?DT>?<JJ.?>*<NN.?.?>+}'  # optional DETERMINER, zero or more ADJECTIVES and one or more NOUNS


# Fill in the other 4 rules (see Part 3 instructions)
hearst_patterns = [
    ('((NP_\w+ ?(, )?)+(and | or )?other NP_\w+)', 'after'),
    ('(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?include (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)', 'before')


]


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: list of tuples. [(sen_token_list, lemmatized_token_list), (.., ..)]
# sen_token is the list of sen token and lemmatized_token is a list of lemmatized token.
def load_corpus(path):
    result = []
    with open(path, "r") as f:
        result = [(line.strip().split("\t")[0].split(" "), line.strip().split("\t")[1].split(" ")) for line in f]
    return result


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: tuple of sets
def load_test(path):
    true_set = set()
    false_set = set()
    with open(path, "r") as f:
        for line in f:
            print(line)
            hyponym = str(line.split("\t")[0]).strip()
            hypernym = str(line.split("\t")[1]).strip()
            status = str(line.split("\t")[2]).strip().lower()
            if status == "true":
                true_set.add((hyponym, hypernym))
            elif status == "false":
                false_set.add((hyponym, hypernym))
    return true_set, false_set


# Fill in the function (see Part 2 instructions)
# Argument type: sentence, lemmatized - list of strings; parser - nltk.RegexpParser
# Return type: string
def chunk_lemmatized_sentence(sentence, lemmatized, parser):
    print("Sentence: " + str(sentence))
    print("Lemmatized: " + str(lemmatized))
    tagged_sen = nltk.pos_tag(sentence)
    print("tagged_sen: " + str(tagged_sen))
    tag_list = [tag for _, tag in tagged_sen]
    print("tag_list: " + str(tag_list))
    tagged_lemmatized = list(zip(lemmatized, tag_list))  # create list of list of tuples.
    print("tag_lemma: " + str(tagged_lemmatized))
    tree = parser.parse(tagged_lemmatized)
    print("tree: " + str(tree))
    # print(tree.draw())
    chunks_list = tree_to_chunks(tree)
    print("chunks_list: " + str(chunks_list))
    merged_chunks = merge_chunks(chunks_list)
    print("merged chunks: " + merged_chunks)
    return merged_chunks


# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: tree - nltk.Tree
# Return type: list of strings
def tree_to_chunks(tree):
    chunks = []
    for child in tree:
        print("child: " + str(child))
        if not isinstance(child, nltk.Tree):
            token, _ = child
            chunks.append(token)
        else:
            print("childs: " + str(child))
            temp = []
            for token, tag in child:
                # for token, tag in grand_child:
                print("grand_child: " + str(token))
                    # token, _ = grand_child
                temp.append(token)
            # temp = [token for l in child.subtrees() for token, tag in l]
            print("temp: " + str(temp))
            chunks.append("NP_" + str("_".join(temp)))
    return chunks


# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: chunks - list of strings
# Return type: string
def merge_chunks(chunks):
    buffer = []
    if chunks is not []:
        buffer.append(chunks[0])
    else:
        return buffer
    for index in range(1, len(chunks)):
        if buffer[-1].startswith("NP_") and chunks[index].startswith("NP_"):
            buffer[-1] = buffer[-1] + str(chunks[index].replace("NP", ""))
        else:
            buffer.append(chunks[index])
    return " ".join(buffer)


# Fill in the function (see Part 4 instructions)
# Argument type: chunked_sentence - string
# Yield type: tuple
def extract_relations(chunked_sentence):
    print("Chunked sentence: " + str(chunked_sentence))
    for pattern, position in hearst_patterns:
        res = re.search(pattern, chunked_sentence)
        if res:
            print("Passed pattern: " + str(pattern))
            match = res.group(0)
            print("match: " + str(match))
            tokens_list = match.split(" ")
            print("tokens_list: " + str(tokens_list))
            NPs = [token for token in tokens_list if token.startswith("NP_")]
            print("NPs: " + str(NPs))
            temp = postprocess_NPs(NPs)
            print("temp: " + str(temp))
            if position == 'before':
                hypernym = temp[0].strip().lower()
                hyponym = temp[1:].strip().lower()
            if position == 'after':
                hypernym = temp[-1].strip().lower()
                hyponym = temp[:-1].strip.lower()
            for x in hyponym:
                yield x, hypernym


# Fill in the function (see Part 4 instructions)
# Helper function for extract_relations()
# Argument type: list of strings
# Return type: list of strings
def postprocess_NPs(NPs):
    t1 = [token.replace("NP_", "").lower() for token in NPs]
    result = []
    # s = ""
    for token in t1:
        result.append(token.replace("_", " "))
        # s = s + token.replace("_", " ") + " "
    # list = s.split(" ")
    # return list[: -1]
    return result


# Fill in the function (see Part 5 instructions)
# Argument type: extractions, gold_true, gold_false - set of tuples
# Return type: tuple
def evaluate_extractions(extractions, gold_true, gold_false):

    tp, fp, fn, flag = 0, 0, 0, False

    for pred_hyper, pred_hypo in extractions:
        for actual_hyper, actual_hypo in gold_true:
            if pred_hyper.lower() == actual_hyper.lower() and pred_hypo.lower() == actual_hypo.lower():
                tp = tp + 1
                flag = True
                break
        if not flag:
            for actual_hyper, actual_hypo in gold_false:
                if pred_hyper.lower() == actual_hyper.lower() and pred_hypo.lower() == actual_hypo.lower():
                    fp = fp + 1
                    break
        flag = False

    fn = len(gold_true) - len(extractions)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * precision * recall) / (precision + recall)

    return precision, recall, f_measure


def main(args):
    corpus_path = args[0]
    test_path = args[1]

    wikipedia_corpus = load_corpus(corpus_path)
    print(str(wikipedia_corpus[0]))
    test_true, test_false = load_test(test_path)

    NP_chunker = nltk.RegexpParser(NP_grammar)

    # Complete the line (see Part 2 instructions)
    # call chunk_lemmatized_sentence() for each sentence and lemmatized list
    wikipedia_corpus = [chunk_lemmatized_sentence(lem_sen_tup[0], lem_sen_tup[1], NP_chunker) for lem_sen_tup in wikipedia_corpus]  #it's a list of strings.
    print("Wikipedia Corpus: " + str(wikipedia_corpus[0]))
    # wikipedia_corpus = ["such NP_authors as NP_Herrick , NP_Goldsmith , and NP_Shakespeare"]
    extracted_pairs = set()
    for chunked_sentence in wikipedia_corpus:
        for pair in extract_relations(chunked_sentence):
            extracted_pairs.add(pair)

    print("extracted_pairs: " + str(extracted_pairs))

    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_extractions(extracted_pairs, test_true, test_false))


if __name__ == '__main__':
    # corpus = 'wikipedia_sentences.txt'
    corpus = 'wiki_mini.txt'
    test = 'test.tsv'
    l = [corpus, test]
    sys.exit(main(l))
    # sys.exit(main(sys.argv[1:]))
