import numpy as np
import os
import pandas as pd

from actinet.utils.sleep_utils import add_sleep_sedentary_transitions


class HMM:
    """
    Implement a basic hidden Markov model (HMM) with parameter saving/loading
    https://en.wikipedia.org/wiki/Hidden_Markov_model.
    """

    def __init__(
        self,
        prior=None,
        emission=None,
        transition=None,
        labels=None,
        uniform_prior=True,
        ignore_transition_gaps=False,
        handle_sleep_transitions=False
    ):
        self.prior = prior
        self.emission = emission
        self.transition = transition
        self.labels = labels
        self.uniform_prior = uniform_prior
        self.ignore_transition_gaps = ignore_transition_gaps
        self.handle_sleep_transitions = handle_sleep_transitions

    def __str__(self):
        return (
            "Hidden Markov Model\n"
            "Ignore transition gaps: {self.ignore_transition_gaps}\n"
            "Correct sleep transitions: {self.handle_sleep_transitions}\n"
            "prior: {self.prior}\n"
            "emission: {self.emission}\n"
            "transition: {self.transition}".format(self=self)
        )

    def fit(self, Y_prob, Y_true, groups=None, T=None, interval=None):
        """
        Fit a HMM to the provided data by calculating the prior, transition and emission matrices.

        :param Y_prob: Observation probabilities
        :type Y_prob: numpy.ndarray
        :param Y_true: Ground truth labels
        :type Y_true: numpy.ndarray
        :param groups: Group labels for the data, if applicable
        :type groups: numpy.ndarray, optional
        :param T: Time at each observation
        :type T: numpy.ndarray, optional
        :param interval: Expected time interval between observations in seconds
        :type interval: int or float, optional
        """

        if self.labels is None:
            self.labels = np.unique(Y_true)

        prior = np.mean(Y_true.reshape(-1, 1) == self.labels, axis=0)

        emission = np.vstack(
            [np.mean(Y_prob[Y_true == label], axis=0) for label in self.labels]
        )

        transition = calculate_transition_matrix(Y_true, groups, T, interval, self.ignore_transition_gaps, 
                                                 self.handle_sleep_transitions)

        self.prior = prior
        self.emission = emission
        self.transition = transition

    def predict(self, y_obs, t=None, interval=None, uniform_prior=None):
        """
        Predict sequence of activities using viterbi algorithm, while restoring labels after gaps in data.

        :param y_obs: Predicted observations
        :type y_obs: numpy.ndarray
        :param t: Time at each observation
        :type t: numpy.ndarray, optional
        :param interval: Expected time interval between observations in seconds
        :type interval: int or float, optional
        :param uniform_prior: Assume uniform priors.
        :type uniform_prior: bool, optional

        :return: Smoothed sequence of activities
        :rtype: np.ndarray
        """
        check_for_input_errors(y_obs, t, interval,
                               ignore_transition_gaps=self.ignore_transition_gaps,
                               handle_sleep_transitions=False)

        y_smooth = self.viterbi(y_obs, uniform_prior)

        if not self.ignore_transition_gaps:
            y_smooth = restore_labels_after_gaps(y_obs, y_smooth, t, interval)

        return y_smooth

    def viterbi(self, y_obs, uniform_prior=None):
        """Perform HMM smoothing over observations via Viteri algorithm
        https://en.wikipedia.org/wiki/Viterbi_algorithm.

        :param y_obs: Predicted observations
        :type y_obs: numpy.ndarray
        :param uniform_prior: Assume uniform priors
        :type uniform_prior: bool, optional

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

        :param path: npz file location
        :type path: str
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            prior=self.prior,
            emission=self.emission,
            transition=self.transition,
            labels=self.labels,
            ignore_transition_gaps=self.ignore_transition_gaps,
            handle_sleep_transitions=self.handle_sleep_transitions
        )

    def load(self, path):
        """
        Load model parameters from a Numpy npz file.

        :param path: npz file location
        :type path: str
        """
        d = np.load(path, allow_pickle=True)
        self.prior = d["prior"]
        self.emission = d["emission"]
        self.transition = d["transition"]
        self.labels = d["labels"]
        self.ignore_transition_gaps = d["ignore_transition_gaps"]
        self.handle_sleep_transitions = d["handle_sleep_transitions"]

    def display(self, labels, precision=3):
        """
        Print the model parameters in a readable format.

        :param labels: List of label names in expected order
        :type labels: list/dict
        :param precision: Number of decimal places to print
        :type precision: int
        """

        pretty_hmm_params(self, labels, precision)


def check_for_input_errors(Y, T, interval, groups=None, ignore_transition_gaps=False, handle_sleep_transitions=False):
    if not ignore_transition_gaps:
        if len(Y) != len(T):
            raise Exception("Provided times should have same length as labels")
        if not interval:
            raise Exception("A window length must be provided when using label times to train hmm")

    if handle_sleep_transitions:
        if len(Y) != len(groups):
            raise Exception("Provided group labels should have same length as labels")


def restore_labels_after_gaps(y_pred, y_smooth, t, interval):
    df = pd.DataFrame({"y_pred": y_pred, "y_smooth": y_smooth})

    if type(t[0]) == int:
        gaps = pd.Series(t).diff(periods=1) != interval
        gaps[0] = False
    else:
        gaps = pd.Series(t).diff(periods=1) != pd.Timedelta(seconds=interval)
        gaps[0] = False

    df.loc[gaps, "y_smooth"] = df.loc[gaps, "y_pred"]

    return df["y_smooth"].values


def calculate_transition_matrix(Y, groups=None, t=None, interval=None,
                                 ignore_transition_gaps=False, handle_sleep_transitions=False):
    check_for_input_errors(Y, t, interval, groups, ignore_transition_gaps, handle_sleep_transitions)

    if ignore_transition_gaps:
        t = range(len(Y))
        interval = 1

    df = pd.DataFrame({
        "label": Y,
        "group": groups
    })

    # create a new column with data shifted one space
    df["shift"] = df["label"].shift(-1)

    # only consider transitions of expected interval
    if type(t[0]) == int:
        df = df[(-pd.Series(t).diff(periods=-1) == interval)]
    else:
        df = df[(-pd.Series(t).diff(periods=-1) == pd.Timedelta(seconds=interval))]

    # correct sleep transitions if required
    if handle_sleep_transitions:
        df = add_sleep_sedentary_transitions(df)

    # add a count column (for group by function)
    df["count"] = 1

    # groupby and then unstack, fill the zeros
    trans_mat = df.groupby(["label", "shift"])["count"].count().unstack().fillna(0)

    # normalise by occurences and save values to get the transition matrix
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values

    if trans_mat.size == 0:
        raise Exception("No transitions found in data")

    return trans_mat


def get_activity_label_code(label, labels):
    try:
        return list(sorted(labels)).index(label)
    except ValueError:
        raise ValueError(f"Label '{label}' not recognised. Must be one of {list(sorted(labels))}.")
    

def print_array(arr, precision=3):
    """Prints all elements of a NumPy array to N decimal places."""
    arr = np.array(arr)
    with np.printoptions(precision=precision, suppress=True):
        print(arr)


def reorder_matrix(data, index_order):
    """Reorder a 1D or 2D square array"""
    arr = np.array(data)

    if arr.ndim == 1:
        return arr[index_order]
    
    elif arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("2D input must be a square matrix.")
        # Reorder rows and columns using the same index order
        return arr[index_order][:, index_order]
    
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def pretty_hmm_params(hmm: HMM, labels, precision=3):
    """Print the HMM parameters in a readable format, reordering them according to the provided labels."""

    if isinstance(labels, dict):
        labels = list(labels.keys())

    elif not isinstance(labels, list):
        raise ValueError("labels must be a list or dict")

    index_order = [get_activity_label_code(label, labels) for label in labels]

    print(f"HMM Parameters ordered {', '.join(labels)}:\n")

    print("Prior:")
    prior = reorder_matrix(hmm.prior, index_order)
    print_array(prior, precision)

    print("\nEmission:")
    emission = reorder_matrix(hmm.emission, index_order)
    print_array(emission, precision)

    print("\nTransition:")
    transition = reorder_matrix(hmm.transition, index_order)
    print_array(transition, precision)
