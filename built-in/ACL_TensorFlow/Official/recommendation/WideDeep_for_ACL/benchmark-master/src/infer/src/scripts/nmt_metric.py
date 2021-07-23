# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import sys
import re
import collections
import math

__all__ = ["evaluate"]


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Compute BLEU score of translated segment against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def evaluate(ref_file, trans_file, metric, subword_option=None):
    """Pick a metric and evaluate depending on task."""
    # BLEU scores for translation task
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, trans_file,
                                 subword_option=subword_option)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence, subword_option):
    """Clean and handle BPE or SPM outputs."""
    sentence = sentence.strip()

    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)

    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

    return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, subword_option=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False
    trans_file_read = open(trans_file, "rb")  # change
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        ref_file_read = open(reference_filename, "rb")  # change
        with codecs.getreader("utf-8")(
                ref_file_read) as fh:  # change
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(trans_file_read) as fh:
        for line in fh:
            line = _clean(line, subword_option=None)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


if __name__ == '__main__':
    ref_file = sys.argv[1]
    trans_file = sys.argv[2]
    score = evaluate(ref_file, trans_file, "bleu", subword_option=None)
    print("bleu:", round(score, 2))
