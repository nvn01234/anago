import json
from unicodedata import category

def should_ignore(word):
    return all(category(c)[0] in ["N", "P"] for c in word)


def update(kb_words, sentences, min_count=0):
    """

    :param kb_words: kb_words.json hiện tại
    :param sentences:   [
                            [(w1, t1), (w2, t2), ...],   # sentence 1
                            ,...
                        ]
    :param min_count:
    :return: kb_words mới
    """
    counters = {}
    new_words = []
    kb_words = {k: v[:] for k, v in kb_words.items()}  # copy
    for sen in sentences:
        for (word, tag), (prev_word, prev_tag) in zip(sen[1:], sen[:-1]):
            if tag not in ['O', '<PAD>'] and prev_tag == 'O' and not should_ignore(prev_word):
                _tag = tag.split("-")[-1]
                if _tag not in counters:
                    counters[_tag] = {}
                if prev_word not in counters[_tag]:
                    counters[_tag][prev_word] = 0
                counters[_tag][prev_word] += 1
    for tag, words_count in counters.items():
        for word, count in words_count.items():
            if count > min_count and word not in kb_words[tag]:
                new_words.append(word)
                kb_words[tag].append(word)
    print("add %d words" % len(new_words))
    lifelong_log = {
        "kb_words": kb_words,
        "counters": counters,
        "new_words": new_words,
    }
    json.dump(lifelong_log, open("log/lifelong_log.json", "w", encoding="utf8"))
    return kb_words