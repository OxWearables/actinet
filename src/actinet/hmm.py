import numpy as np
import os
import pandas as pd


class HMM:
    """
    Implement a basic HMM model with parameter saving/loading.
    """

    def __init__(
        self,
        prior=None,
        emission=None,
        transition=None,
        labels=None,
        uniform_prior=True,
    ):
        self.prior = prior
        self.emission = emission
        self.transition = transition
        self.labels = labels
        self.uniform_prior = uniform_prior

    def __str__(self):
        return (
            "Hidden Markov Model\n"
            "prior: {prior}\n"
            "emission: {emission}\n"
            "transition: {transition}".format(
                prior=self.prior,
                emission=self.emission,
                transition=self.transition,
            )
        )

    def fit(self, Y_prob, Y_true, T=None, interval=None):
        """https://en.wikipedia.org/wiki/Hidden_Markov_model
        :param Y_prob: Observation probabilities
        :param Y_true: Ground truth labels
        """

        if self.labels is None:
            self.labels = np.unique(Y_true)

        prior = np.mean(Y_true.reshape(-1, 1) == self.labels, axis=0)

        emission = np.vstack(
            [np.mean(Y_prob[Y_true == label], axis=0) for label in self.labels]
        )

        transition = calculate_transition_matrix(Y_true, T, interval)

        self.prior = prior
        self.emission = emission
        self.transition = transition

    def predict(self, y_obs, t=None, interval=None, uniform_prior=None):
        check_for_time_values_error(y_obs, t, interval)

        y_smooth = self.viterbi(y_obs, uniform_prior)

        if t is not None:
            y_smooth = restore_labels_after_gaps(y_obs, y_smooth, t, interval)

        return y_smooth

    def viterbi(self, y_obs, uniform_prior=None):
        """Perform HMM smoothing over observations via Viteri algorithm
        https://en.wikipedia.org/wiki/Viterbi_algorithm
        :param y_obs: Predicted observation
        :param bool uniform_prior: Assume uniform priors.

        :return: Smoothed sequence of activities
        :rtype: np.ndarray
        """

        def log(x):
            return np.log(x + 1e-16)

        prior = (
            np.ones(len(self.labels)) / len(self.labels)
            if (self.uniform_prior or uniform_prior)
            else self.prior
        )
        emission = self.emission
        transition = self.transition
        labels = self.labels

        nobs = len(y_obs)
        n_labels = len(labels)

        y_obs = np.where(y_obs.reshape(-1, 1) == labels)[1]  # to numeric

        probs = np.zeros((nobs, n_labels))
        probs[0, :] = log(prior) + log(emission[:, y_obs[0]])
        for j in range(1, nobs):
            for i in range(n_labels):
                probs[j, i] = np.max(
                    log(emission[i, y_obs[j]]) + log(transition[:, i]) + probs[j - 1, :]
                )  # probs already in log scale
        viterbi_path = np.zeros_like(y_obs)
        viterbi_path[-1] = np.argmax(probs[-1, :])
        for j in reversed(range(nobs - 1)):
            viterbi_path[j] = np.argmax(
                log(transition[:, viterbi_path[j + 1]]) + probs[j, :]
            )  # probs already in log scale

        viterbi_path = labels[viterbi_path]  # to labels

        return viterbi_path

    def save(self, path):
        """
        Save model parameters to a Numpy npz file.

        :param str path: npz file location
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            prior=self.prior,
            emission=self.emission,
            transition=self.transition,
            labels=self.labels,
        )

    def load(self, path):
        """
        Load model parameters from a Numpy npz file.

        :param str path: npz file location
        """
        d = np.load(path, allow_pickle=True)
        self.prior = d["prior"]
        self.emission = d["emission"]
        self.transition = d["transition"]
        self.labels = d["labels"]


def check_for_time_values_error(Y, T, interval):
    # If truthy, T must be the same length as Y, and interval must also be truthy
    if T is not None:
        if len(Y) != len(T):
            raise Exception("Provided times should have same length as labels")
        if not interval:
            raise Exception(
                "A window length must be provided when using label times to train hmm"
            )


def restore_labels_after_gaps(y_pred, y_smooth, t, interval):
    # Restore unsmoothed predictions to labels following gaps in time
    df = pd.DataFrame({"y_pred": y_pred, "y_smooth": y_smooth})

    if type(t[0]) == int:
        gaps = pd.Series(t).diff(periods=1) != interval
        gaps[0] = False
    else:
        gaps = pd.Series(t).diff(periods=1) != pd.Timedelta(seconds=interval)
        gaps[0] = False

    df.loc[gaps, "y_smooth"] = df.loc[gaps, "y_pred"]

    return df["y_smooth"].values


def calculate_transition_matrix(Y, t=None, interval=None):
    # t and interval are used to identify any gaps in the data
    # If not provided, it is assumed there are no gaps
    check_for_time_values_error(Y, t, interval)

    t = t if t is not None else range(len(Y))
    interval = interval or 1

    df = pd.DataFrame(Y)

    # create a new column with data shifted one space
    df["shift"] = df[0].shift(-1)

    # only consider transitions of expected interval
    if type(t[0]) == int:
        df = df[(-pd.Series(t).diff(periods=-1) == interval)]
    else:
        df = df[(-pd.Series(t).diff(periods=-1) == pd.Timedelta(seconds=interval))]

    # add a count column (for group by function)
    df["count"] = 1

    # groupby and then unstack, fill the zeros
    trans_mat = df.groupby([0, "shift"]).count().unstack().fillna(0)

    # normalise by occurences and save values to get the transition matrix
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values

    if trans_mat.size == 0:
        raise Exception("No transitions found in data")

    return trans_mat
