from sklearn.metrics import f1_score
import numpy as np

def get_baseline_f1_score(average='weighted') -> float | np.ndarray[float]:
    '''
    Returns a baseline weighted F1-score for a dumb model always predicting the majority class

    Args:
    - average: see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
      None: one F1-score for each class
      'weighted' (default): weighted average of F1-scores (more influenced by performance of majority class)
      'macro': unweighted mean of all F1-scores
      'micro': compute a global single F1-score as if there was only one class to predict

    Results:
      - [0., 0.74375235, 0.] for None
      - 0.440 for weighted,
      - 0.248 for macro,
      - 0.592 for micro.
    '''
    nb_bothcells = 3448

    nb_healthy = 28895
    nb_rubbish = 50371
    nb_unhealthy = 2366 + nb_bothcells

    nb_total = nb_healthy + nb_rubbish + nb_unhealthy

    healthy = [1, 0, 0]
    rubbish = [0, 1, 0]
    unhealthy = [0, 0, 1]

    y_true = np.vstack([
        np.tile(healthy, (nb_healthy, 1)),
        np.tile(rubbish, (nb_rubbish, 1)),
        np.tile(unhealthy, (nb_unhealthy, 1))
    ])

    y_pred = np.tile(rubbish, (nb_total, 1)) # 0.440

    # if y_pred was predicting "randomly" the 2 majority classes
    # y_pred = np.tile([rubbish, healthy], (nb_total//2, 1)) # 0.458
    # y_pred = np.tile([rubbish]*5 + [healthy]*3, (nb_total//8, 1)) # 0.481

    return f1_score(y_true, y_pred, average=average)
