from collections import Counter
from itertools import combinations
import math
import pandas as pd
from typing import Mapping, Sequence, Any, Tuple, NewType, cast, Optional

distribution = NewType('distribution', Mapping[Any, float])


def get_sample_df() -> pd.DataFrame:
    """
    Define a sample dataframe that we can re-use in docstring tests.
    """
    columns = ['gender', 'age', 'eyes']
    data = [[0, 30, 1], [0, 35, 1], [1, 30, 2], [1, 30, 1], [1, 35, 2], [1, 35, 2], [1, 30, 2], [0, 30, 1]]
    return pd.DataFrame(data, columns=columns)


def calculate_kl(posterior: distribution,
                 prior: distribution,
                 ) -> float:
    """Calculate D_KL(P || Q) (the KL-divergence) in bits.

    D_KL(P || Q) is the `information gained when one revises one's
    beliefs from the prior probability distribution Q to the posterior
    probability distribution P`. (Wikipedia, Kullback-Leibler divergence)

    `p` and `q` are both dictionaries mapping some hashable to a number.
    It is assumed that they are both normalised: their values should add
    up to 0. `q` must not have any 0 values unless the corresponding `p`
    value is also 0.

    >>> calculate_kl(posterior={0: 1, 1: 0}, prior={0: 0.5, 1: 0.5})
    1.0
    >>> calculate_kl(posterior={0: 0.25, 1: 0.75}, prior={0: 0.5, 1: 0.5})
    0.18872187554086717
    """
    return sum(pk * math.log2(pk / prior[k])
               for k, pk in posterior.items()
               if pk > 0)


def calculate_distribution_from_values(values: Sequence) -> distribution:
    """
    >>> assert calculate_distribution_from_values([1, 2, 3, 2, 3, 2, 2, 3]) == {1: 0.125, 2: 0.5, 3: 0.375}
    >>> assert calculate_distribution_from_values([(1, 2), (2, 2), (3, 2), (3, 2), (2, 2)]) == \
    {(1, 2): 0.2, (3, 2): 0.4, (2, 2): 0.4}
    """
    counts = Counter(values)
    total_counts = sum(counts.values())
    return cast(distribution, {v: c / total_counts for v, c in counts.items()})


# def calculate_dataframe_column_priors(df: pd.DataFrame) -> Mapping[str, distribution]:
#     """
#     >>> df = get_sample_df()
#     >>> assert calculate_dataframe_column_priors(df) == {
#     ...     'gender': {0: 0.375, 1: 0.625}, 'age': {30: 0.625, 35: 0.375}, 'eyes': {1: 0.5, 2: 0.5}}
#     """
#     return {column_name: calculate_distribution_from_values(list(df[column_name]))
#             for column_name in df}
#

def as_sequence_of_tuples(df: pd.DataFrame) -> Sequence[Tuple]:
    """
    >>> as_sequence_of_tuples(get_sample_df())
    ((0, 30, 1), (0, 35, 1), (1, 30, 2), (1, 30, 1), (1, 35, 2), (1, 35, 2), (1, 30, 2), (0, 30, 1))
    """
    return tuple(tuple(row) for row in df.values)


def calculate_prior(df: pd.DataFrame,
                    column_names: Sequence[str],
                    ) -> distribution:
    """
    Calculate a single prior distribution for the combination of columns provided.
    >>> df = get_sample_df()
    >>> calculate_prior(df, ['gender'])
    {(0,): 0.375, (1,): 0.625}
    >>> assert calculate_prior(df, ['gender', 'age']) == {(0, 30): 0.25, (0, 35): 0.125, (1, 35): 0.25, (1, 30): 0.375}
    """
    return calculate_distribution_from_values(as_sequence_of_tuples(df[column_names]))


def calculate_posterior(df: pd.DataFrame,
                        given: Mapping[str, Any],
                        ) -> distribution:
    """
    Given the values of some columns, what is the distribution of the remaining columns in the data frame?
    >>> df = get_sample_df()
    >>> calculate_posterior(df, {'gender': 1})
    {(30, 2): 0.4, (30, 1): 0.2, (35, 2): 0.4}
    >>> calculate_posterior(df, {'gender': 0, 'age': 30})
    {(1,): 1.0}
    """
    column_names = list(given.keys())
    column_values = tuple(given[c] for c in column_names)
    unknown_column_names = [c for c in df if c not in column_names]
    # Using isin(a single element list) temporarily until I can find a working "equality" function.
    restricted_df = df[df[column_names].apply(tuple, axis=1).isin([column_values])][unknown_column_names]
    return calculate_distribution_from_values(as_sequence_of_tuples(restricted_df))


def calculate_kl_if_attacker_knows_some_values(df: pd.DataFrame,
                                               known: Mapping[str, Any],
                                               prior: Optional[distribution] = None,
                                               ) -> float:
    """
    If an attacker knows some attributes of a person, what is their information gain on seeing the full data?
    (Assumes the attacker knows the person is in the dataset.)
    The prior is on the attributes the attacker doesn't already know.
    If not provided explicitly, assume the prior is just the distribution in the data.
    >>> df = get_sample_df()
    >>> calculate_kl_if_attacker_knows_some_values(df, {'gender': 1})
    0.3610794049684064
    >>> calculate_kl_if_attacker_knows_some_values(df, {'gender': 0, 'age': 30})
    1.0
    """
    unknown_column_names = [c for c in df if c not in known]
    if prior is None:
        prior = calculate_prior(df, column_names=unknown_column_names)
    posterior = calculate_posterior(df, given=known)
    return calculate_kl(posterior=posterior, prior=prior)


def calculate_kls_if_attacker_knows_columns(df: pd.DataFrame,
                                            known_column_names: Sequence[str],
                                            prior: Optional[distribution] = None,
                                            ) -> Mapping[Tuple, float]:
    """
    Returns the KL divergences if the attacker already has a subset of the data,
    containing all the rows but only these column names.
    Returns a mapping from the known values to the KL divergence.
    >>> df = get_sample_df()
    >>> calculate_kls_if_attacker_knows_columns(df, ['gender'])
    {(0,): 1.0250624987980728, (1,): 0.3610794049684064}
    >>> calculate_kls_if_attacker_knows_columns(df, ('gender', 'age'))
    {(0, 30): 1.0, (0, 35): 1.0, (1, 35): 1.0, (1, 30): 0.08170416594551039}
    """
    known_column_names = list(known_column_names)  # pandas treats lists and tuples differently.
    unknown_column_names = [c for c in df if c not in known_column_names]
    if not prior:
        # More efficient to calculate this once upfront.
        prior = calculate_prior(df, column_names=unknown_column_names)

    all_values_of_known_columns = set(as_sequence_of_tuples(df[known_column_names]))
    known_value_mappings = [{k: v for k, v in zip(known_column_names, values)}
                           for values in all_values_of_known_columns]
    return {tuple(known_value_mapping.values()):
                calculate_kl_if_attacker_knows_some_values(df, known=known_value_mapping, prior=prior)
            for known_value_mapping in known_value_mappings}


def calculate_kls_for_attackers(df: pd.DataFrame,
                                numbers_of_known_columns: Sequence[int] = None,
                                ) -> Mapping[Sequence[str], Mapping[Tuple, float]]:
    """
    Returns the KL divergences for all possible subsets of columns with the given lengths.

    >>> df = get_sample_df()
    >>> calculate_kls_for_attackers(df, [1])
    {('gender',): {(0,): 1.0250624987980728, (1,): 0.3610794049684064}, ('age',): {(35,): 0.22004999903845832, (30,): 0.04408690482417521}, ('eyes',): {(2,): 0.7075187496394219, (1,): 0.603759374819711}}
    >>> calculate_kls_for_attackers(df)
    {('gender',): {(0,): 1.0250624987980728, (1,): 0.3610794049684064}, ('age',): {(35,): 0.22004999903845832, (30,): 0.04408690482417521}, ('eyes',): {(2,): 0.7075187496394219, (1,): 0.603759374819711}, ('gender', 'age'): {(0, 30): 1.0, (0, 35): 1.0, (1, 35): 1.0, (1, 30): 0.08170416594551039}, ('gender', 'eyes'): {(0, 1): 0.005431269113550144, (1, 1): 0.6780719051126377, (1, 2): 0.04655470219574073}, ('age', 'eyes'): {(30, 1): 0.2510864671689521, (35, 1): 1.4150374992788437, (35, 2): 0.6780719051126377, (30, 2): 0.6780719051126377}}
    """
    column_names = [c for c in df]
    if numbers_of_known_columns is None:
        numbers_of_known_columns = list(range(1, len(df.columns)))
    result = {}
    for number_of_known_columns in numbers_of_known_columns:
        for known_column_names in combinations(column_names, number_of_known_columns):
            result[known_column_names] = calculate_kls_if_attacker_knows_columns(df, known_column_names)
    return result

    # return {known_column_names: calculate_kls_if_attacker_knows_columns(df, known_column_names)
    #         for number_of_known_columns in numbers_of_known_columns
    #         for known_column_names in combinations(column_names, number_of_known_columns)
    #         }


def get_map_of_totals(map_of_maps: Mapping[Sequence[str], Mapping[Tuple, float]]) -> Mapping[Sequence[str], float]:
    """
    >>> df = get_sample_df()
    >>> get_map_of_totals(calculate_kls_for_attackers(df, [1]))
    {('gender',): 1.3861419037664793, ('age',): 0.26413690386263355, ('eyes',): 1.311278124459133}
    >>> get_map_of_totals(calculate_kls_for_attackers(df))
    {('gender',): 1.3861419037664793, ('age',): 0.26413690386263355, ('eyes',): 1.311278124459133, ('gender', 'age'): 3.0817041659455104, ('gender', 'eyes'): 0.7300578764219287, ('age', 'eyes'): 3.0222677766730714}
    """
    return {column_names: sum(value_map.values()) for column_names, value_map in map_of_maps.items()}


def get_max_total(map_of_maps: Mapping[Sequence[str], Mapping[Tuple, float]]) -> float:
    """
    >>> df = get_sample_df()
    >>> get_max_total(calculate_kls_for_attackers(df, [1]))
    1.3861419037664793
    >>> get_max_total(calculate_kls_for_attackers(df))
    3.0817041659455104
    """
    return max(get_map_of_totals(map_of_maps).values())
