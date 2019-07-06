import numpy as np
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from typing import List

chencherry = SmoothingFunction()


def evaluate_ppl(total_loss: float, n_predict_words: int):
    """ Calculate perplexity
    Args:
        n_predict_word (int): The total number of words to predict
        total_loss (float): The total loss of whole set
    """
    ppl = np.exp(total_loss / n_predict_words)
    return ppl


def evaluate_bleu(
    true_sequence: List[str], predicted_sequence: List[str]
) -> float:
    score = nltk.translate.bleu_score.sentence_bleu(
        [true_sequence],
        predicted_sequence,
        smoothing_function=chencherry.method4,
    )
    return score
