import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))




import pandas as pd
import numpy as np
import itertools


def filter_by_year(df, year_column, period):
    """Filter the DataFrame for a specific year."""
    return df[df[year_column] == period].copy()

def apply_filters(df, filters=None):
    """Apply column-specific filters to the DataFrame."""
    if filters:
        for column, value in filters.items():
            df = df[df[column] == value]
    return df.fillna(0).reset_index(drop=True)

def calculate_overall_kpi(df, kpi, calculation):
    """Calculate the aggregated KPI for a given DataFrame."""
    return df[kpi].agg(calculation)

def calculate_kpi_changes(current_kpi, previous_kpi):
    """Calculate absolute and percentage changes between two KPIs."""
    abs_change = current_kpi - previous_kpi
    perc_change = 100 * (abs_change / previous_kpi if previous_kpi != 0 else 0)
    return abs_change, perc_change

def calculate_grouped_kpi_changes(current_df, previous_df, kpi, grouping, calculation):
    """Calculate KPI changes for specific groupings."""
    current_agg = current_df.groupby(grouping)[kpi].agg(calculation).reset_index()
    current_agg.rename(columns={kpi: 'kpi_current'}, inplace=True)

    previous_agg = previous_df.groupby(grouping)[kpi].agg(calculation).reset_index()
    previous_agg.rename(columns={kpi: 'kpi_previous'}, inplace=True)

    merged = pd.merge(current_agg, previous_agg, on=grouping, how='outer').fillna(0)
    merged['abs_change'], merged['perc_change'] = zip(
        *merged.apply(
            lambda row: calculate_kpi_changes(row['kpi_current'], row['kpi_previous']), axis=1
        )
    )
    return merged

def filter_and_sort_by_threshold(df, threshold, sort_by='perc_change', ascending=False):
    """Filter rows by percentage change threshold and sort."""
    filtered_df = df[np.abs(df['perc_change']) >= threshold].copy()
    return filtered_df.sort_values(by=sort_by, ascending=ascending)

def calculate_ratio(df, numerator_col, denominator_col):
    """Calculate the ratio between two columns."""
    total_numerator = df[numerator_col].sum()
    total_denominator = df[denominator_col].sum()
    ratio = total_numerator / total_denominator if total_denominator != 0 else None
    return ratio, total_numerator


def perform_root_cause_analysis(
    df, current_period, previous_period, kpi, levels, calculation='sum', thresholds=None, service_filter=None
):
    """Orchestrates the root cause analysis."""
    if thresholds is None:
        thresholds = [0] * len(levels)

    # Prepare data for each period
    current_df = apply_filters(filter_by_year(df, 'Year', current_period), service_filter)
    previous_df = apply_filters(filter_by_year(df, 'Year', previous_period), service_filter)

    # Calculate overall change
    current_kpi = calculate_overall_kpi(current_df, kpi, calculation)
    previous_kpi = calculate_overall_kpi(previous_df, kpi, calculation)
    abs_change, perc_change = calculate_kpi_changes(current_kpi, previous_kpi)
    results = [{"kpi_current": current_kpi, "kpi_previous": previous_kpi, "abs_change": abs_change,
                "perc_change": perc_change, "depth_level": "L0"}]

    # Perform grouped analysis
    for perm in itertools.permutations(levels):
        temp_current_df, temp_previous_df = current_df.copy(), previous_df.copy()

        for i, level in enumerate(perm):
            grouping = list(perm[:i + 1])
            grouped_changes = calculate_grouped_kpi_changes(temp_current_df, temp_previous_df, kpi, grouping, calculation)
            
            filtered_grouped_changes = filter_and_sort_by_threshold(grouped_changes, thresholds[i])
            if not filtered_grouped_changes.empty:
                filtered_grouped_changes['depth_level'] = f"L{i+1}"
                filtered_grouped_changes['permutation'] = str(perm)
                results.extend(filtered_grouped_changes.to_dict(orient='records'))

                temp_current_df = temp_current_df[temp_current_df[level].isin(filtered_grouped_changes[level])]
                temp_previous_df = temp_previous_df[temp_previous_df[level].isin(filtered_grouped_changes[level])]
            else:
                break

    # Convert results to dictionary format
    results_df = pd.DataFrame(results)
    return results_df
