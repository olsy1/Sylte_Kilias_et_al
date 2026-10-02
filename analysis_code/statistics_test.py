
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools
import warnings
from itertools import combinations



def _display(obj):
    try:
        from IPython.display import display as _ipython_display
        _ipython_display(obj)
    except Exception:
        print(obj)


def _display_header(text):
    try:
        from IPython.display import display as _ipython_display, HTML
        _ipython_display(HTML(f"<b>{text}</b>"))
    except Exception:
        print(text)


def animal_labels(datapaths):
    return [str(p).replace('\\', '/').split('/')[-1].split('.')[0] for p in datapaths]


def day_labels(days, prefix='d'):
    return [f'{prefix}{d}' for d in days]


def _labels(labels, n, prefix):
    if labels is None:
        return [f'{prefix}{i + 1}' for i in range(n)]
    labels = [str(x) for x in labels]
    if len(labels) != n:
        warnings.warn(f'got {len(labels)} {prefix} labels for {n} levels - '
                      f'check the labels passed to the test')
        labels = (labels + [f'{prefix}{i + 1}' for i in range(len(labels), n)])[:n]
    return labels


def _is_nested(labels):
    return (labels is not None and len(labels) > 0
            and isinstance(labels[0], (list, tuple, np.ndarray)))


def _col_labels_per_condition(col_labels, n_conditions):
    if _is_nested(col_labels):
        per = list(col_labels)
        if len(per) != n_conditions:
            warnings.warn(f'got {len(per)} column-label lists for {n_conditions} conditions')
            per = (per + [None] * n_conditions)[:n_conditions]
        return per
    return [col_labels] * n_conditions


def _shared_col_labels(col_labels):
    if not _is_nested(col_labels):
        return col_labels
    per_condition = [list(c) for c in col_labels]
    shared = []
    for values in zip(*per_condition):
        unique = list(dict.fromkeys(str(v) for v in values))
        shared.append(unique[0] if len(unique) == 1 else ' | '.join(unique))
    return shared


def data_table(array, row_labels=None, col_labels=None,
               row_name='animal', col_name='day', decimals=4):
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    df = pd.DataFrame(
        arr,
        index=pd.Index(_labels(row_labels, arr.shape[0], 'animal'), name=row_name),
        columns=pd.Index(_labels(col_labels, arr.shape[1], 'level'), name=col_name),
    )
    return df if decimals is None else df.round(decimals)


def print_data(arrays, row_labels=None, col_labels=None, condition_labels=None,
               title=None, row_name='animal', col_name='day', decimals=4,
               summary=False):
    conditions = list(arrays) if isinstance(arrays, (list, tuple)) else [arrays]
    conditions = [np.atleast_2d(np.asarray(a, dtype=float)) for a in conditions]

    names = _labels(condition_labels, len(conditions), 'condition')
    cols = _col_labels_per_condition(col_labels, len(conditions))

    if title is not None:
        _display_header(title)

    for arr, name, col in zip(conditions, names, cols):
        _display_header(f'datapoints{" - " + name if name else ""}  '
                        f'({arr.shape[0]} {row_name}s x {arr.shape[1]} {col_name}s)')
        df = data_table(arr, row_labels=row_labels, col_labels=col,
                        row_name=row_name, col_name=col_name, decimals=decimals)
        _display(df)
        if summary:
            stats_df = pd.DataFrame(
                {'mean': df.mean(axis=0),
                 'sem': df.std(axis=0, ddof=1) / np.sqrt(len(df)),
                 'n': df.notna().sum(axis=0)}
            ).T
            _display(stats_df if decimals is None else stats_df.round(decimals))


def fit_mixed_model(data_dict, standardize=True, row_labels=None, obs_labels=None,
                    var_names=None, title=None, show_data=True, decimals=4,
                    row_name='animal', col_name='observation'):
    from statsmodels.formula.api import mixedlm
    from sklearn.preprocessing import StandardScaler
    
    data_dict = data_dict.copy()

    if 'animal_id' not in data_dict:
        animal_ids = []
        for i, x_array in enumerate(data_dict['x1']):
            animal_ids.append(np.full(len(x_array), i+1))
        data_dict['animal_id'] = animal_ids
    
    x_vars = [key for key in data_dict.keys() if key.startswith('x')]
    
    data_cols = {}
    for key in data_dict:
        if key in ['y', 'animal_id'] or key.startswith('x'):
            data_cols[key] = np.hstack(data_dict[key])
    
    data = pd.DataFrame(data_cols)

    if show_data:
        _print_mixed_model_data(data_dict, data, x_vars, row_labels=row_labels,
                                obs_labels=obs_labels, var_names=var_names,
                                title=title, decimals=decimals,
                                row_name=row_name, col_name=col_name)

    if standardize:
        scaler = StandardScaler()
        std_vars = []
        for x_var in x_vars:
            std_name = f'{x_var}_std'
            data[std_name] = scaler.fit_transform(data[[x_var]])
            std_vars.append(std_name)
            
        data['y_std'] = StandardScaler().fit_transform(data[['y']])
        formula = "y_std ~ " + " + ".join(std_vars)
        prefix = 'std'
        re_vars = std_vars  #
    else:
        formula = "y ~ " + " + ".join(x_vars)
        prefix = ''
        re_vars = x_vars 
    re_formula = "~" + " + ".join(re_vars)
    model = mixedlm(formula, data, groups=data["animal_id"])
    #model = mixedlm(formula, data, groups=data["animal_id"], re_formula=re_formula)
    results = model.fit()
    
    return results, data, prefix


def _print_mixed_model_data(data_dict, data, x_vars, row_labels=None, obs_labels=None,
                            var_names=None, title=None, decimals=4,
                            row_name='animal', col_name='observation'):
    groups = list(data_dict['x1'])
    group_sizes = [len(np.asarray(g).ravel()) for g in groups]
    animals = _labels(row_labels, len(groups), 'animal')

    long = pd.DataFrame({
        row_name: np.concatenate([[a] * n for a, n in zip(animals, group_sizes)]),
        col_name: np.concatenate([_labels(obs_labels, n, 'obs') for n in group_sizes]),
    })
    renames = dict(zip(x_vars + ['y'], var_names)) if var_names else {}
    for var in x_vars + ['y']:
        long[renames.get(var, var)] = data[var].to_numpy()

    if title is not None:
        _display_header(title)
    _display_header(f'datapoints - mixed model ({len(long)} observations from '
                    f'{len(animals)} {row_name}s)')
    _display(long if decimals is None else long.round(decimals))

    if len(set(group_sizes)) == 1:
        n_obs = group_sizes[0]
        for var in x_vars + ['y']:
            name = renames.get(var, var)
            _display_header(f'datapoints - {name} ({row_name} x {col_name})')
            _display(data_table(data[var].to_numpy().reshape(len(animals), n_obs),
                                row_labels=animals, col_labels=obs_labels,
                                row_name=row_name, col_name=col_name,
                                decimals=decimals))


def mixedlm(arrays, row_labels=None, title=None, show_data=True, decimals=4):
    """
    mixed linear model
    """
    if show_data:
        names = _labels(row_labels, len(arrays), 'animal')
        if title is not None:
            _display_header(title)
        _display_header('datapoints - mixed linear model')
        long = pd.DataFrame({
            'animal': np.concatenate([[n] * len(a) for n, a in zip(names, arrays)]),
            'value': np.concatenate([np.asarray(a, dtype=float) for a in arrays]),
        })
        _display(long if decimals is None else long.round(decimals))

    mouse =np.hstack([np.ones(len(arr)) * i for i, arr in enumerate(arrays)])
    values = np.hstack(arrays)

    values = (values - np.mean(values)) / np.std(values)

    df = pd.DataFrame({'mouse': mouse, 'values': values})

    md = smf.mixedlm("values ~ 1", df, groups=df["mouse"])
    mdf = md.fit(method=["lbfgs"])
    return mdf.summary()


def repeated_measures_anova_general(arrays, row_labels=None, col_labels=None,
                                    condition_labels=None, title=None,
                                    row_name='animal', col_name='day',
                                    show_data=True, decimals=4):
    """
    repeated measures anova for any number of conditions
    """
    n_conditions = len(arrays)
    n_subjects, n_timepoints = arrays[0].shape

    for arr in arrays[1:]:
        assert arr.shape == (n_subjects, n_timepoints), "All arrays must have the same shape"

    if show_data:
        print_data(arrays, row_labels=row_labels, col_labels=col_labels,
                   condition_labels=condition_labels,
                   title=title if title is not None else 'two-way RM ANOVA',
                   row_name=row_name, col_name=col_name, decimals=decimals)

    data = []
    for condition, array in enumerate(arrays):
        for subject in range(n_subjects):
            for time in range(n_timepoints):
                data.append({
                    'subject': subject,
                    'time': time,
                    'condition': condition,
                    'value': array[subject, time]
                })
   
    df = pd.DataFrame(data)
   
    aov_results = pg.rm_anova(data=df, dv='value', within=['time', 'condition'], 
                             subject='subject', detailed=True)

    post_hoc_df = post_hoc_repeated_measures(arrays, col_labels=col_labels,
                                             condition_labels=condition_labels,
                                             show_data=False)

    return aov_results, post_hoc_df
def repeated_measures_anova_single_condition(array, row_labels=None, col_labels=None,
                                             condition_label=None, title=None,
                                             row_name='animal', col_name='day',
                                             show_data=True, decimals=4):
    """
    repeated measures anova for a single condition across time
    """
    n_subjects, n_timepoints = array.shape

    if show_data:
        print_data([array], row_labels=row_labels, col_labels=col_labels,
                   condition_labels=[condition_label if condition_label else ''],
                   title=title if title is not None else 'one-way RM ANOVA',
                   row_name=row_name, col_name=col_name, decimals=decimals)

    data = []
    for subject in range(n_subjects):
        for time in range(n_timepoints):
            data.append({
                'subject': subject,
                'time': time,
                'value': array[subject, time]
            })
    
    df = pd.DataFrame(data)

    aov_results = pg.rm_anova(data=df, dv='value', within=['time'], subject='subject', detailed=True)
    
    return aov_results



def two_way_repeated_measures_anova(data1, data2, **kwargs):
    return repeated_measures_anova_general([data1, data2], **kwargs)

def general_anova(arrays, group_labels=None, title=None, show_data=True, decimals=4):
    """
    one-way ANOVA for any number of groups.
    """
    if show_data:
        names = _labels(group_labels, len(arrays), 'group')
        if title is not None:
            _display_header(title)
        _display_header('datapoints - one-way ANOVA (between groups)')
        long = pd.DataFrame({
            'group': np.concatenate([[n] * len(a) for n, a in zip(names, arrays)]),
            'value': np.concatenate([np.asarray(a, dtype=float) for a in arrays]),
        })
        _display(long if decimals is None else long.round(decimals))

    groups =[np.full(len(arr), i) for i, arr in enumerate(arrays)]
    values = np.concatenate(arrays)
    groups = np.concatenate(groups)
    df = pd.DataFrame({'group': groups, 'value': values})

    return pg.anova(data=df, dv='value', between='group', detailed=True)

def check_normality_and_equality_of_variances(arrays):
    _, p_shapiro = stats.shapiro(np.concatenate(arrays))
    _, p_levene = stats.levene(*arrays)
    return p_shapiro > 0.05 and p_levene > 0.05

def post_hoc_non_paired(arrays, method='tukey', p_adjust='bonferroni',
                        group_labels=None, title=None, show_data=True, decimals=4):
    """
    Perform post-hoc tests for non-paired data.

    """
    if show_data:
        names = _labels(group_labels, len(arrays), 'group')
        if title is not None:
            _display_header(title)
        _display_header('datapoints - post-hoc (non-paired)')
        long = pd.DataFrame({
            'group': np.concatenate([[n] * len(a) for n, a in zip(names, arrays)]),
            'value': np.concatenate([np.asarray(a, dtype=float) for a in arrays]),
        })
        _display(long if decimals is None else long.round(decimals))

    groups =[np.full(len(arr), i) for i, arr in enumerate(arrays)]
    values = np.concatenate(arrays)
    groups = np.concatenate(groups)
    df = pd.DataFrame({'group': groups, 'value': values})

    if method == 'tukey':
        return pairwise_tukeyhsd(values, groups)
    elif method in ['t-test', 'mann-whitney']:
        return pg.pairwise_tests(data=df, dv='value', between='group', 
                                 parametric=(method == 't-test'), 
                                 padjust=p_adjust)
    else:
        raise ValueError("Method must be 'tukey', 't-test', or 'mann-whitney'")



def post_hoc_repeated_measures(arrays, p_adjust='bonferroni', alpha=0.05,
                               row_labels=None, col_labels=None,
                               condition_labels=None, title=None,
                               row_name='animal', col_name='day',
                               show_data=True, decimals=4):
    """
    post-hoc analysis for repeated measures data with automatic test selection.

    """
    n_conditions = len(arrays)
    n_subjects, n_timepoints = arrays[0].shape

    for arr in arrays[1:]:
        if arr.shape != (n_subjects, n_timepoints):
            raise ValueError("All arrays must have the same shape")

    if show_data:
        print_data(arrays, row_labels=row_labels, col_labels=col_labels,
                   condition_labels=condition_labels,
                   title=title if title is not None else 'post-hoc (repeated measures)',
                   row_name=row_name, col_name=col_name, decimals=decimals)

    time_names = _labels(_shared_col_labels(col_labels), n_timepoints, 'T')
    condition_names = _labels(condition_labels, n_conditions, 'C')

    results_list = []
    all_p_values = []
    
    for t in range(n_timepoints):
        for i, j in combinations(range(n_conditions), 2):
            data1 = arrays[i][:, t]
            data2 = arrays[j][:, t]
            
            differences = data1 - data2
            _, normality_p = stats.shapiro(differences)
            
            if normality_p > alpha:  
                statistic, p_value = stats.ttest_rel(data1, data2)
                test_name = 't-test'
            else:  
                statistic, p_value = stats.wilcoxon(data1, data2)
                test_name = 'wilcoxon'

            all_p_values.append(p_value)
            
            if test_name == 't-test':
                effect_size = np.mean(differences) / np.std(differences, ddof=1)
                effect_size_name = "Cohen's d"
            else:
                n = len(differences)
                z = np.abs(statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                effect_size = z / np.sqrt(n)
                effect_size_name = 'r'

            results_list.append({
                'Timepoint': time_names[t],
                'Condition 1': condition_names[i],
                'Condition 2': condition_names[j],
                'Test Used': test_name,
                'Statistic': round(statistic, 4),
                'p-value': p_value,
                'Normal Dist.': normality_p > alpha,
                'Normality p-value': round(normality_p, 4),
                'Effect Size': round(effect_size, 4),
                'Effect Size Type': effect_size_name,
                'Mean Cond 1': round(np.mean(data1), 4),
                'Mean Cond 2': round(np.mean(data2), 4),
                'Mean Diff': round(np.mean(differences), 4),
                'SD Diff': round(np.std(differences, ddof=1), 4),
                'N': len(data1)
            })
    
    df = pd.DataFrame(results_list)
    
    adjusted_p_values = pg.multicomp(all_p_values, method=p_adjust)[1]
    df[f'Adjusted p-value ({p_adjust})'] = adjusted_p_values
    
    df['Significance'] = ''
    df.loc[df['Adjusted p-value ({})'.format(p_adjust)] <= 0.001, 'Significance'] = '***'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.001) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.01), 'Significance'] = '**'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.01) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.05), 'Significance'] = '*'
    
    for col in ['p-value', f'Adjusted p-value ({p_adjust})', 'Normality p-value']:
        df[col] = df[col].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
    
    column_order = [
        'Timepoint', 'Condition 1', 'Condition 2',
        'Test Used', 'Statistic', 
        'p-value', f'Adjusted p-value ({p_adjust})', 'Significance',
        'Effect Size', 'Effect Size Type', 
        'Mean Cond 1', 'Mean Cond 2', 'Mean Diff', 'SD Diff',
        'N', 'Normal Dist.', 'Normality p-value'
    ]
    df = df[column_order]
    
    return df

def post_hoc_timepoints(arrays, p_adjust='bonferroni', alpha=0.05,
                        row_labels=None, col_labels=None, title=None,
                        row_name='animal', col_name='day',
                        show_data=True, decimals=4):
    """
    post-hoc tests for paired data across all possible timepoint combinations.est
    """
    if len(arrays) < 2:
        raise ValueError("At least two timepoints are required for comparison")

    timepoint_pairs = list(itertools.combinations(range(len(arrays)), 2))

    if show_data:
        print_data([np.asarray(arrays, dtype=float).T],
                   row_labels=row_labels, col_labels=col_labels,
                   condition_labels=[''],
                   title=title if title is not None else 'post-hoc (paired, across timepoints)',
                   row_name=row_name, col_name=col_name, decimals=decimals)

    time_names = _labels(col_labels, len(arrays), 'T')

    results_list = []
    all_p_values = []
   
    for i, (t1_idx, t2_idx) in enumerate(timepoint_pairs):
        differences = arrays[t1_idx] - arrays[t2_idx]

        _, normality_p = stats.shapiro(differences)

        if normality_p > alpha:  
            statistic, p_value = stats.ttest_rel(arrays[t1_idx], arrays[t2_idx])
            test_name = 't-test'
        else:  
            statistic, p_value = stats.wilcoxon(arrays[t1_idx], arrays[t2_idx])
            test_name = 'wilcoxon'
       
        all_p_values.append(p_value)
        
        if test_name == 't-test':
            effect_size = np.mean(differences) / np.std(differences, ddof=1)
            effect_size_name = "Cohen's d"
        else:
            n = len(differences)
            z = np.abs(statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            effect_size = z / np.sqrt(n)
            effect_size_name = 'r'
        
        results_list.append({
            'Timepoint 1': time_names[t1_idx],
            'Timepoint 2': time_names[t2_idx],
            'Test Used': test_name,
            'Statistic': round(statistic, 4),
            'p-value': p_value,
            'Normal Dist.': normality_p > alpha,
            'Normality p-value': round(normality_p, 4),
            'Effect Size': round(effect_size, 4),
            'Effect Size Type': effect_size_name,
            'Mean Diff': round(np.mean(differences), 4),
            'SD Diff': round(np.std(differences, ddof=1), 4)
        })
   
    df = pd.DataFrame(results_list)
    
    adjusted_p_values = pg.multicomp(all_p_values, method=p_adjust)[1]
    df[f'Adjusted p-value ({p_adjust})'] = adjusted_p_values
    
    df['Significance'] = ''
    df.loc[df['Adjusted p-value ({})'.format(p_adjust)] <= 0.001, 'Significance'] = '***'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.001) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.01), 'Significance'] = '**'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.01) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.05), 'Significance'] = '*'
    
    for col in ['p-value', f'Adjusted p-value ({p_adjust})', 'Normality p-value']:
        df[col] = df[col].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
    
    column_order = [
        'Timepoint 1', 'Timepoint 2', 'Test Used', 'Statistic', 
        'p-value', f'Adjusted p-value ({p_adjust})', 'Significance',
        'Effect Size', 'Effect Size Type', 'Mean Diff', 'SD Diff',
        'Normal Dist.', 'Normality p-value'
    ]
    df = df[column_order]
    
    return df
def post_hoc_paired_multiple(arrays1, arrays2, p_adjust='bonferroni', alpha=0.05,
                             row_labels=None, col_labels=None,
                             condition_labels=None, title=None,
                             row_name='animal', col_name='comparison',
                             show_data=True, decimals=4):
    """
    post-hoc tests for paired data across multiple comparisons
    """
    if len(arrays1) != len(arrays2):
        raise ValueError("arrays1 and arrays2 must have the same length")

    if show_data:
        print_data([np.asarray(arrays1, dtype=float).T, np.asarray(arrays2, dtype=float).T],
                   row_labels=row_labels, col_labels=col_labels,
                   condition_labels=condition_labels,
                   title=title if title is not None else 'post-hoc (paired, multiple comparisons)',
                   row_name=row_name, col_name=col_name, decimals=decimals)

    comparison_names = _labels(_shared_col_labels(col_labels), len(arrays1), 'Comp ')

    results_list = []
    all_p_values = []
    
    for i, (arr1, arr2) in enumerate(zip(arrays1, arrays2)):
        differences = arr1 - arr2
        
        _, normality_p = stats.shapiro(differences)

        if normality_p > alpha:  
            statistic, p_value = stats.ttest_rel(arr1, arr2)
            test_name = 't-test'
        else: 
            statistic, p_value = stats.wilcoxon(arr1, arr2)
            test_name = 'wilcoxon'
        
        all_p_values.append(p_value)
      
        if test_name == 't-test':
            effect_size = np.mean(differences) / np.std(differences, ddof=1)
            effect_size_name = "Cohen's d"
        else:
            n = len(differences)
            z = np.abs(statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            effect_size = z / np.sqrt(n)
            effect_size_name = 'r'
        
        results_list.append({
            'Comparison': comparison_names[i],
            'Test Used': test_name,
            'Statistic': round(statistic, 4),
            'p-value': p_value,
            'Normal Dist.': normality_p > alpha,
            'Normality p-value': round(normality_p, 4),
            'Effect Size': round(effect_size, 4),
            'Effect Size Type': effect_size_name,
            'Mean Group 1': round(np.mean(arr1), 4),
            'Mean Group 2': round(np.mean(arr2), 4),
            'Mean Diff': round(np.mean(differences), 4),
            'SD Diff': round(np.std(differences, ddof=1), 4),
            'N': len(arr1)
        })
    
    df = pd.DataFrame(results_list)
    
    adjusted_p_values = pg.multicomp(all_p_values, method=p_adjust)[1]
    df[f'Adjusted p-value ({p_adjust})'] = adjusted_p_values
    
    df['Significance'] = ''
    df.loc[df['Adjusted p-value ({})'.format(p_adjust)] <= 0.001, 'Significance'] = '***'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.001) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.01), 'Significance'] = '**'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.01) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.05), 'Significance'] = '*'
    
    for col in ['p-value', f'Adjusted p-value ({p_adjust})', 'Normality p-value']:
        df[col] = df[col].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
    
    column_order = [
        'Comparison', 'Test Used', 'Statistic', 
        'p-value', f'Adjusted p-value ({p_adjust})', 'Significance',
        'Effect Size', 'Effect Size Type', 
        'Mean Group 1', 'Mean Group 2', 'Mean Diff', 'SD Diff',
        'N', 'Normal Dist.', 'Normality p-value'
    ]
    df = df[column_order]
    
    return df



