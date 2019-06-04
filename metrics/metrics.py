import numpy as np


def evaluate_ppl(total_loss: float, n_predict_words: int):
    """ Calculate perplexity
    Args:
        n_predict_word (int): The total number of words to predict
        total_loss (float): The total loss of whole set
    """
    ppl = np.exp(total_loss / n_predict_words)
    return ppl
