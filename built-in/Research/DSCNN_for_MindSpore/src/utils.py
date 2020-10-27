
SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'
def prepare_words_list(wanted_words):
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words