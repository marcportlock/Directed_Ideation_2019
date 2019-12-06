import collections
import itertools
import math
import operator
import random

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

skip_feature = lambda x: None  # Aggregation function skipping a feature
no_aggregation = lambda x: x  # No-op aggregation function


def calculate_kl(p, q):
    """Calculate D_KL(P || Q) (the KL-divergence) in bits.
    
    D_KL(P || Q) is the `information gained when one revises one's
    beliefs from the prior probability distribution Q to the posterior
    probability distribution P`. (Wikipedia, Kullback–Leibler
    divergence)

    `p` and `q` are both dictionaries mapping some hashable to a number.
    It is assumed that they are both normalised: their values should add
    up to 0. `q` must not have any 0 values unless the corresponding `p`
    value is also 0.
    """
    return sum(pk * math.log2(pk / q[k])
               for k, pk in p.items()
               if pk > 0)


def calculate_distribution(values, accuracy=1, feature_distribution=None):
    """Calculate the distribution of the values.

    If `accuracy` == 1, this function simply counts the values and
    normalises their counts into probabilities.

    If `accuracy` < 1, then it mixes the counts with the
    `feature_distribution`, which you can think of as a prior for the
    feature. We assume that we draw from `values` `accuracy` of the time
    and from `feature_distribution` 1 - `accuracy` of the time.

    The case where `accuracy` < 1 is slow.
    """
    assert accuracy == 1 or feature_distribution is not None
    counts = collections.Counter(values)
    total_counts = sum(counts.values())
    if accuracy == 1:
        return {v: c / total_counts for v, c in counts.items()}
    else:
        assert counts.keys() <= feature_distribution.keys()
        acc_tc = accuracy / total_counts
        onemacc = 1 - accuracy
        probs = ((v, counts[v] * acc_tc + p * onemacc)
                 for v, p in feature_distribution.items())
        return {v: p for v, p in probs if p > 0}


def find_kls_for_features(
        dataset, feature_is, feature_distributions, accuracies):
    """Find the KL divergence of feature values against the prior.

    We find the true distribution of the features taking into account
    the accuracy. We then compute the KL divergence.
    """
    l = len(dataset[0])
    assert all(len(row) == l for row in dataset)

    # one bucket per set of 'known' features
    buckets = [collections.defaultdict(list) for _ in range(len(feature_is))]
    bucket_map = [[] for _ in range(len(feature_is))]  ########
    for row in dataset:
        key = tuple(row[i] for i in range(l) if i not in feature_is)
        for i, j in enumerate(feature_is):
            buckets[i][key].append(row[j])
            bucket_map[i].append(key)

    bucket_kls = [
        {
            key: calculate_kl(
                calculate_distribution(bucket,
                                       accuracy=accuracies[feature_is[i]],
                                       feature_distribution=feature_distributions[feature_is[i]]),
                feature_distributions[feature_is[i]])
            for key, bucket in feature_buckets.items()}
        for i, feature_buckets in enumerate(buckets)]

    return [tuple(map(bucket_kls[i].__getitem__, bucket_map[i]))
            for i in range(len(feature_is))]



def aggregate(dataset, aggregation):
    """Perform aggregation with a set of aggregation functions.

    You can think of it as a per-field `map`. Every field of the dataset
    is transformed using an aggregation function if one is specified.

    `dataset` is a list of lists of values. `aggregation` is a
    dictionary mapping the feature index to a transformation function. A
    feature whose index is missing from `aggregation` is left alone.

    Example: We have a dataset of 5 rows with name and postcode.
    dataset = [
        ['John',    '3426'],
        ['Jim',     '2174'],
        ['Jane',    '3472'],
        ['Jannine', '2163'],
        ['Jack',    '7461']
    ]

    We wish to aggregate by only keeping the first 2 digits of the
    postcode. Then aggregation = {
        1 : lambda postcode: postcode[:2]
    }. We call aggregate(dataset, aggregation) and obtain:
    [
        ['John',    '34'],
        ['Jim',     '21'],
        ['Jane',    '34'],
        ['Jannine', '21'],
        ['Jack',    '74']
    ]    
    """
    aggregation = dict(aggregation)
    l = len(dataset[0])
    assert all(len(row) == l for row in dataset)
    assert all(i in range(l) for i in aggregation)
    for i in range(l):
        if i not in aggregation:
            aggregation[i] = no_aggregation

    return [[f(row[i]) for i, f in sorted(aggregation.items())]
            for row in dataset]


def is_valid_prior(prior_distribution, feature_values):
    """Check that the assumed prior is valid for the feature values.

    `prior_distribution` is valid if:
    - every value if `feature_values` has a nonzero probability,
    - every value in `prior_distribution` is non-negative, and
    - the values in `prior_distribution` add up to 1 (up to floating-
      -point errors).
    """
    if not all(val in fd for val in feature_values):
        return False
    if not all(fd[val] > 0 for val in feature_values):
        return False
    if not all(v >= 0 for v in fd.values()):
        return False
    if not math.isclose(sum(fd.values()), 1):
        return False
    return True


def C(n, r):
    return math.factorial(n) // math.factorial(n - r) // math.factorial(r)


def sample_is(n, r, samples):
    if samples is None:
        yield from itertools.combinations(range(n), r)
    else:
        total_combinations = C(n, r)
        if samples > total_combinations:
            raise ValueError('more samples than combinations')
        if samples >= total_combinations >> 1:
            all_combinations = list(itertools.combinations(range(n), r))
            random.shuffle(all_combinations)

            num_produced = 0
            feature_produced = [False] * n
            for i, comb in enumerate(all_combinations):
                if num_produced >= samples:
                    break
                if all(map(feature_produced.__getitem__, comb)):
                    continue
                for j in comb:
                    feature_produced[j] = True
                num_produced += 1
                all_combinations[i] = None
                yield comb

            for comb in all_combinations:
                if num_produced >= samples:
                    break
                if comb is not None:
                    yield comb
        else:
            already_produced = set()
            feature_produced = [False] * n
            while len(already_produced) < samples:
                comb = random.sample(range(n), r)
                comb = tuple(sorted(comb))
                if (comb not in already_produced
                        and (all(already_produced)
                             or not all(map(already_produced.__getitem__,
                                            comb)))):
                    already_produced.add(comb)
                    for i in comb:
                        feature_produced[i] = True
                    yield comb


def find_risks_for_records(aggregated_dataset,
                           feature_priors={},
                           feature_accuracies={},
                           unknown_features=1,
                           samples=None):
    """Find the risk (as KL divergence from prior) for all attributes.

    `feature_priors` are optional. It is a dictionary mapping the
    feature index to an assumed prior. If not provided, the prior for
    the feature is calculated from the global distribution.

    `feature_accuracies` maps the feature index to the accuracy of the
    feature. If not provided for a feature, it defaults to 1.
    """
    l = len(aggregated_dataset[0])
    assert all(len(row) == l for row in aggregated_dataset)

    feature_priors = feature_priors.copy()
    for i in range(l):
        if i in feature_priors:
            fd = feature_priors[i]
        else:
            ffs = tuple(map(operator.itemgetter(i), aggregated_dataset))
            fd = calculate_distribution(ffs)
        feature_priors[i] = fd

    feature_accuracies = feature_accuracies.copy()
    for i in range(l):
        if i not in feature_accuracies:
            feature_accuracies[i] = 1

    feature_counts = [0] * l
    feature_kls = [[0] * len(aggregated_dataset) for _ in range(l)]
    for is_ in sample_is(l, unknown_features, samples):
        feature_kls_this = find_kls_for_features(
            aggregated_dataset,
            is_,
            feature_priors,
            feature_accuracies)
        for i, feature_kl in zip(is_, feature_kls_this):
            feature_kl_previous = feature_kls[i]
            feature_kls[i] = tuple(map(
                operator.add, feature_kl, feature_kl_previous))

        for i in is_:
            feature_counts[i] += 1

    for i, denom in enumerate(feature_counts):
        feature_kls[i] = tuple(map(
            operator.truediv,
            feature_kls[i],
            itertools.repeat(denom)))

    return list(zip(*feature_kls))


def display_risks(header, risks, rows=20, minr=0, maxr=10):
    """Return an object that prints the attribute-wise risks in Jupyter.

    We colour-code the risks for nice visuals.

    `header` is a list that specifies the table headers.

    `risks` is a list of lists of floats.

    We print at most `rows` (default 20) rows.

    `minr` (default 0) and `maxr` (default 10) specify the cutoff points
    for our colour coding. Anything above `maxr` will be coloured deep
    red for very risky. `minr` is similar.

    The code below is magic and I don't remember how it works.
    """
    df = pd.DataFrame(risks, columns=header)[:rows]
    cm = matplotlib.colors.ListedColormap(
        sns.color_palette("RdYlGn", 256).as_hex()[::-1])
    def risk_to_color(risk):
        color_i = round((min(risk, maxr) - minr) * 255 / (maxr - minr))
        color = cm.colors[color_i]
        return f'background-color: {color}'
    s = df.style.applymap(risk_to_color)
    return s

def find_individual_risks(risks):
    """Sum the attribute-wise risks per row to get individual risks.

    `risks` is a list of lists of floats. The result is one list of
    floats.
    """
    irisks = list(map(sum, risks))
    return irisks


def plot_individual_risk_cdf(irisks):
    """Plots the cumulative histogram of individual risks.

    `irisks` are individual risks as a list of floats.
    """
    irisks = sorted(irisks, reverse=True)
    y, x = zip(*enumerate(irisks, start=1))
    y += len(irisks),
    x += 0,
    plt.plot(x, y)
    plt.xlabel('Risk (bits)')
    plt.ylabel('Count')
    plt.title('Risk CDF')
    plt.xlim((0, 50))


def percentile(individual_risks, p):
    """Find the `p`-th percentile of `individual_risks`.

    We use this to calculate PIF_95 and PIF_99.
    """
    return np.percentile(individual_risks, p)


