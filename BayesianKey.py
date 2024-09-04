from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re 

app = Flask(__name__)

# Load the character matrix
character_matrix = pd.read_excel('character_matrix.xlsx', index_col=0)
num_species = character_matrix.shape[0]
global_prior = np.ones(num_species) / num_species

similarity_matrices = {
    'Numbers of dorsal spines on tibia I, II, III, IV: tibial spine formula': pd.DataFrame({
        '0000': [1, 0.5, 0.2, 0.1, 0.05, 0.02],
        '0011': [0.5, 1, 0.5, 0.2, 0.1, 0.05],
        '1111': [0.2, 0.5, 1, 0.5, 0.2, 0.1],
        '2211': [0.1, 0.2, 0.5, 1, 0.5, 0.2],
        '2221': [0.05, 0.1, 0.2, 0.5, 1, 0.5],
        '2222': [0.02, 0.05, 0.1, 0.2, 0.5, 1]
    }, index=['0000', '0011', '1111', '2211', '2221', '2222']),
    'Length of prosoma by range [mm]' : pd.DataFrame({
        '0.4-0.6': [1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545, 0.416667, 0.384615, 0.357143, 0.333333, 0.3125, 0.294118],
    '0.6-0.8': [0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545, 0.416667, 0.384615, 0.357143, 0.333333, 0.3125],
    '0.8-1.0': [0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545, 0.416667, 0.384615, 0.357143, 0.333333],
    '1.0-1.2': [0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545, 0.416667, 0.384615, 0.357143],
    '1.2-1.4': [0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545, 0.416667, 0.384615],
    '1.4-1.6': [0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545, 0.416667],
    '1.6-1.8': [0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5, 0.454545],
    '1.8-2.0': [0.416667, 0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556, 0.5],
    '2.0-2.2': [0.384615, 0.416667, 0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625, 0.555556],
    '2.2-2.4': [0.357143, 0.384615, 0.416667, 0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286, 0.625],
    '2.4-2.6': [0.333333, 0.357143, 0.384615, 0.416667, 0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333, 0.714286],
    '2.6-2.8': [0.3125, 0.333333, 0.357143, 0.384615, 0.416667, 0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0, 0.833333],
    '2.8-3.0': [0.294118, 0.3125, 0.333333, 0.357143, 0.384615, 0.416667, 0.454545, 0.5, 0.555556, 0.625, 0.714286, 0.833333, 1.0]
}, index=[
    '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.4', '1.4-1.6',
    '1.6-1.8', '1.8-2.0', '2.0-2.2', '2.2-2.4', '2.4-2.6', '2.6-2.8', '2.8-3.0'
]), 
    'Length of femur I : relative to prosoma' : pd.DataFrame({
    'equal in length': [1.0, 0.2, 0.2],  # Similarity with 'Equal in length', 'Longer than prosoma', 'Shorter than prosoma'
    'longer than prosoma': [0.2, 1.0, 0.1],  # Similarity with 'Equal in length', 'Longer than prosoma', 'Shorter than prosoma'
    'shorter than prosoma': [0.2, 0.1, 1.0]  # Similarity with 'Equal in length', 'Longer than prosoma', 'Shorter than prosoma'
}, index=['equal in length', 'longer than prosoma', 'shorter than prosoma']),
    'Prosoma: appearance' : pd.DataFrame({ # NEED TO CHNAGE 
    'inconspicuous': [1.0, 0.3, 0.3, 0.3],  # Similarity with 'Inconspicuous', 'Margin with teeth', 'With conspicuous hairs/spines', 'With pits (dorsally)'
    'margin with teeth': [0.3, 1.0, 0.2, 0.2],  # Similarity with 'Inconspicuous', 'Margin with teeth', 'With conspicuous hairs/spines', 'With pits (dorsally)'
    'with pits (dorsally)': [0.3, 0.1, 1.0, 0.1],  # Similarity with 'Inconspicuous', 'Margin with teeth', 'With conspicuous hairs/spines', 'With pits (dorsally)'
    'with conspicuous hairs/spines': [0.3, 0.3, 0.1, 1.0]  # Similarity with 'Inconspicuous', 'Margin with teeth', 'With conspicuous hairs/spines', 'With pits (dorsally)'
}, index=['inconspicuous', 'margin with teeth', 'with pits (dorsally)', 'with conspicuous hairs/spines']),
    'Opisthosoma: appearance' : pd.DataFrame({
    'unicolorous, inconspicuous': [1.0, 0.3, 0.3, 0.1],
    'patterned': [0.3, 1.0, 0.6, 0.1],
    'with scutum': [0.3, 0.6, 1.0, 0.6],
    'conspicuously hairy': [0.1, 0.1, 0.6, 1.0]
    }, index=['unicolorous, inconspicuous', 'patterned', 'with scutum', 'conspicuously hairy']),
    'Dorsal spines on femur I: count' : pd.DataFrame({
        'multiple': [1.0, 0.3, 0.1],
        'one': [0.3, 1.0, 0.3],
        'none': [0.1, 0.3, 1.0]}, index=['multiple', 'one', 'none']),
    'Posterior eye row: form' : pd.DataFrame({
        'straight': [1.0, 0.3, 0.3],
        'procurved': [0.3, 1.0, 0.1],
        'recurved': [0.3, 0.1, 1.0]}, index=['straight', 'procurved', 'recurved']),
    'Posterior median eye (PME) separation: relative to diameter (d)': pd.DataFrame({
        'distinctly greater than d': [1.0, 0.1, 0.3],
        'distinctly less than d': [0.1, 1.0, 0.3],
        'equal to d': [0.3, 0.3, 1.0]
    }, index=['distinctly greater than d', 'distinctly less than d', 'equal to d']), 
    'Headregion of male: appearance' : pd.DataFrame({
        'inconspicuous': [1.0, 0.2, 0.2, 0.2, 0.2],
        'sulci present': [0.3, 1.0, 0.5, 0.6, 0.1],
        'with lobe (simple)': [0.3, 0.4, 1.0, 0.4, 0.1],
        'complex': [0.2, 0.4, 0.4, 1.0, 0.4],
        'with horns/tufts': [0.3, 0.1, 0.1, 0.1, 1.0]}, index=['inconspicuous', 'sulci present', 'with lobe (simple)', 'complex', 'with horns/tufts']),
    'Overall appearance: Colouring' : pd.DataFrame({
        'inconspicuous, generally dark': [1.0, 0.05, 0.1, 0.3],
        'pale, ground or cave dweller (careful with older specimens: fading)': [0.3, 1.0, 0.3, 0.1],
        'bright, red, orange': [0.1, 0.05, 1.0, 0.3],
        'conspicuous contrast legs/prosoma': [0.05, 0.05, 0.05, 1.0]
    }, index = ['nconspicuous, generally dark', 'pale, ground or cave dweller (careful with older specimens: fading)', 'bright, red, orange', 'conspicuous contrast legs/prosoma']),
    'Position of TmI by range: relative to metatarsus' : pd.DataFrame({
        '0.10-0.19': [1.000000, 0.909091, 0.833333, 0.769231, 0.714286, 0.666667, 0.625000, 0.588235, 0.555556],
    '0.20-0.29': [0.909091, 1.000000, 0.909091, 0.833333, 0.769231, 0.714286, 0.666667, 0.625000, 0.588235],
    '0.30-0.39': [0.833333, 0.909091, 1.000000, 0.909091, 0.833333, 0.769231, 0.714286, 0.666667, 0.625000],
    '0.40-0.49': [0.769231, 0.833333, 0.909091, 1.000000, 0.909091, 0.833333, 0.769231, 0.714286, 0.666667],
    '0.50-0.59': [0.714286, 0.769231, 0.833333, 0.909091, 1.000000, 0.909091, 0.833333, 0.769231, 0.714286],
    '0.60-0.69': [0.666667, 0.714286, 0.769231, 0.833333, 0.909091, 1.000000, 0.909091, 0.833333, 0.769231],
    '0.70-0.79': [0.625000, 0.666667, 0.714286, 0.769231, 0.833333, 0.909091, 1.000000, 0.909091, 0.833333],
    '0.80-0.89': [0.588235, 0.625000, 0.666667, 0.714286, 0.769231, 0.833333, 0.909091, 1.000000, 0.909091],
    '0.90-0.99': [0.555556, 0.588235, 0.625000, 0.666667, 0.714286, 0.769231, 0.833333, 0.909091, 1.000000]
    }, index= ['0.10-0.19', '0.20-0.29', '0.30-0.39', '0.40-0.49', '0.50-0.59',
    '0.60-0.69', '0.70-0.79', '0.80-0.89', '0.90-0.99']),
    'Anterior median eyes: size relative to anterior lateral eyes ALE': pd.DataFrame({
        'distinctly smaller than ALE': [1.0, 0.7, 0.3],
        'about the same as ALE': [0.7, 1.0, 0.7],
        'distinctly larger than ALE': [0.3, 0.7, 1.0]
    }, index = ['distinctly smaller than ALE', 'about the same as ALE', 'distinctly larger than ALE']),
    'Prolateral spines on femur I: count' : pd.DataFrame({
        'multiple': [1.0, 0.3, 0.1],
        'one': [0.3, 1.0, 0.3],
        'none': [0.1, 0.3, 1.0]}, index=['multiple', 'one', 'none']),
    'Prolateral spines on tibia I: count' : pd.DataFrame({
        'multiple': [1.0, 0.3, 0.1],
        'one': [0.3, 1.0, 0.3],
        'none': [0.1, 0.3, 1.0]}, index=['multiple', 'one', 'none']),
    'Conspicuous structures on chelicerae: appearance': pd.DataFrame({
        'none':[1.0, 0.3, 0.3],
        'apophyses/teeth-like prcoesses/tubercles': [0.3, 1.0, 0.7],
        'spines': [0.3, 0.7, 1.0]}, index=['none', 'apophyses/teeth-like processes/tubercles', 'spines']),
    'Sternum: appearance': pd.DataFrame({
        'smooth':[1.0, 0.3, 0.3],
        'rugose': [0.3, 1.0, 0.7],
        'pitted': [0.3, 0.7, 1.0]
    }, index=['smooth', 'rugose', 'pitted']),
    'Width of sternum between coxae IV: relative to width of coxae IV (d)': pd.DataFrame({
        'distinctly greater than d': [1.0, 0.1, 0.3],
        'distinctly less than d': [0.1, 1.0, 0.3],
        'equal to d': [0.3, 0.3, 1.0]
    }, index=['distinctly greater than d', 'distinctly less than d', 'equal to d']),
    'Dorsal spines on metatarsus I: count' : pd.DataFrame({
        'multiple': [1.0, 0.3, 0.1],
        'one': [0.3, 1.0, 0.3],
        'none': [0.1, 0.3, 1.0]}, index=['multiple', 'one', 'none']),
    'Male pedipalp: patella appearance': pd.DataFrame({
        'unremarkable':[1.0, 0.3, 0.3, 0.3],
        'with apophyses': [0.3, 1.0, 0.5, 0.5],
        'with conspicuous spines': [0.3, 0.5, 1.0, 0.5],
        'conspicuously swollen': [0.3, 0.5, 0.5, 1.0]
    }, index=['unremarkable', 'with apophyses', 'with conspicuous spines', 'conspicuously swollen']),
    'Male pedipalp: tibia appearance': pd.DataFrame({
        'unremarkable':[1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        'with simple apophysis':[0.4, 1.0, 0.6, 0.4, 0.4, 0.4, 0.4],
        'with multiple, simple apophyses':[0.4, 0.5, 1.0, 0.5, 0.1, 0.1, 0.1],
        'with complex apophysis':[0.1, 0.1, 0.2, 1.0, 0.6, 0.4, 0.4],
        'with multiple, complex apophyses': [0.4, 0.4, 0.4, 0.5, 1.0, 0.3, 0.3],
        'with one or more spines': [0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 0.7],
        'with tufts of hair or spines':[0.2, 0.2, 0.2, 0.2, 0.2, 0.7, 1.0]
    }, index=['unremarkable','with simple apophysis','with multiple, simple apophyses','with complex apophysis','with multiple, complex apophyses','with one or more spines','with tufts of hair or spines']),
    'Male pedipalp: cymbium appearance': pd.DataFrame({
        'simple':[1.0, 0.5, 0.5],
        'with dorsal projections/conical elevations':[0.5, 1.0, 0.8],
        'margin with notches/bulges': [0.5, 0.8, 1.0]
    }, index=['simple', 'with dorsal projection/conical elevations', 'margin with notches/bulges']),
    'Male pedipalp: embolus appearance': pd.DataFrame({
        'unremarkable':[1.0, 0.5, 0.5],
        'conspicuous, circular':[0.5, 1.0, 0.8],
        'conspicuous, curled':[0.5, 0.8, 1.0]
    }, index=['unremarkable', 'conspicuous, circular', 'conspicuous, curled']),
    'Epigyne: appearance': pd.DataFrame({
        'unremarkable':[1.0, 0.3, 0.3, 0.3, 0.3, 0.3],
        'with atrium/cavity':[0.3, 1.0, 0.3, 0.3, 0.3, 0.3],
        'with septum/medial structure':[0.3, 0.3, 1.0, 0.3, 0.3, 0.3],
        'with lateral plates':[0.3, 0.3, 0.3, 1.0, 0.3, 0.3],
        'with scape (from anterior margin)':[0.3, 0.3, 0.3, 0.3, 1.0, 0.3],
        'with parmula (from posterior margin)':[0.3, 0.3, 0.3, 0.3, 0.3, 1.0]
        }, index=['unremarkable','with atrium/cavity','with septum/medial structure','with lateral plates','with scape (from anterior margin)','with parmula (from posterior margin)'])
}

def value_in_multiple(needle, haystack):
    h = str(haystack)
    n = str(needle)
    value_list = [v.strip() for v in re.split(r'[;|]', h)]  # split the 'needle(value)' from the haystack(all values from that character) by ; or |
    return n in value_list  # Returns True if the needle is found

def compute_similarity_scores(character_matrix, observed_characters, error_rate):
    similarity_scores = np.zeros(character_matrix.shape[0])
    # List of characters for which similarity scores should be computed
    typable_characters = [
        'Length of prosoma [mm]', 
        'Position of trichobothrium on metatarsus I (TmI): relative to metatarsus']

    for character, value in observed_characters.items():
        if character in typable_characters:
            try:
                matrix_values = character_matrix[character].astype(float).to_numpy()
                matrix_values = np.nan_to_num(matrix_values, nan=0.0)  # Replace NaN with zero
                # Convert observed values to float
                if isinstance(value, list):
                    values = [float(v) for v in value]
                else:
                    values = [float(value)]
                
                # Calculate number of unique values for this character
                unique_values_count = len(np.unique(matrix_values))
                user_specified_count = len(values)
                # Compute similarity scores and error distribution
                for v in values:
                    distances = np.abs(matrix_values - v)
                    distances = np.nan_to_num(distances, nan=np.inf)  # Ensure no NaN values in distances
                    
                    # Compute similarity score and error distribution for each species
                    for i in range(len(distances)):
                        if distances[i] == 0:
                            # Exact match: adjust similarity score
                            similarity_scores[i] += 1
                        else:
                            # Non-match: calculate similarity score
                            similarity_scores[i] += 1 / (1 + distances[i])
                    
                # Exact matches should reduce the error rate
                exact_matches = np.isin(matrix_values, values)
                similarity_scores[exact_matches] -= error_rate
                               
                # Non-matches should add the distributed error rate
                non_exact_matches = ~exact_matches
                if unique_values_count > user_specified_count:
                    distributed_error = error_rate / (unique_values_count - user_specified_count)
                    similarity_scores[non_exact_matches] += distributed_error
            
            except ValueError as ve:
                print(f"ValueError: {ve} (Character: {character}, Value: {value})")
            except TypeError as te:
                print(f"TypeError: {te} (Character: {character}, Value: {value})")
    
    # Handle the dummy species
    if 'Not in species list' in character_matrix.index:
        dummy_idx = character_matrix.index.get_loc('Not in species list')
        # Calculate the dummy species likelihood as the difference between the highest and lowest similarity scores
        highest_score = np.max(similarity_scores)
        lowest_score = np.min(similarity_scores[similarity_scores != 0])
        dummy_likelihood = highest_score - lowest_score
        similarity_scores[dummy_idx] = dummy_likelihood
    
    # Normalise the similarity scores
    unique_scores = np.unique(similarity_scores)
    total_likelihood = unique_scores.sum()
    
    if total_likelihood > 0:
        similarity_scores /= total_likelihood
    
    return similarity_scores


def binary_character_likelihood(character_matrix, character, values, error_rate):
    num_species = character_matrix.shape[0]
    likelihood = np.ones(num_species)
    if not isinstance(values, list):
        values = [values]
    condition = character_matrix[character].apply(lambda x: any(val in str(x).split(';') for val in values))
    r = condition.astype(int)
    likelihood *= r
    nan_mask = character_matrix[character].isna()
    unique_values = character_matrix[character].dropna().unique()
    
    # Distribute error rate
    match_likelihood = 1 - error_rate  
    non_match_likelihood = error_rate

    # Assign match and non-match likelihoods
    likelihood = np.where(r == 1, match_likelihood, non_match_likelihood)
    # For species with NaN values, set the likelihood to the non-match likelihood
    likelihood[nan_mask] = non_match_likelihood
    
    if 'Not in species list' in character_matrix.index:
        dummy_idx = character_matrix.index.get_loc('Not in species list')
        likelihood[dummy_idx] = 0.5

    # Normalise likelihoods
    total_likelihood = likelihood.sum()
    if total_likelihood > 0:
        likelihood /= total_likelihood

    return likelihood


def adjust_similarity_matrices(similarity_matrices, error_rate, user_values):
    adjusted_matrices = {}
    original_matrices = {}

    for character, matrix in similarity_matrices.items():
        if not isinstance(matrix, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(matrix)} for character {character}")

        original_matrices[character] = matrix.copy()
        # adjust the matrix for error rates
        unique_values = character_matrix[character].dropna().astype(str).str.split(';').explode().unique()
        n_unique_values = len(unique_values)
        n_user_values = len(user_values.get(character, []))
        # Compute non-match adjustment
        if n_unique_values > n_user_values:
            non_match_adjustment = error_rate / (n_unique_values - n_user_values)
        else:
            non_match_adjustment = error_rate
        # Apply error rate adjustment directly
        adjusted_matrix = matrix.applymap(
            lambda x: x + non_match_adjustment if x < 1 else x - error_rate
        )
        adjusted_matrix = adjusted_matrix.clip(lower=0)
        adjusted_matrices[character] = adjusted_matrix
    return adjusted_matrices, original_matrices


def categorical_character_likelihood(character_matrix, character, values, error_rate):
    num_species = character_matrix.shape[0]
    likelihood = np.ones(num_species)

    adjusted_matrices, original_matrices = adjust_similarity_matrices(similarity_matrices, error_rate, {character: values})
    similarity_matrix = adjusted_matrices.get(character)
    original_matrix = original_matrices.get(character)
    
    if similarity_matrix is None or original_matrix is None:
        raise ValueError(f"Similarity matrix not defined for character: {character}")

    # Convert single values to list if needed
    if not isinstance(values, list):
        values = [values]
    
    similarity_scores = np.zeros(num_species)
    
    # Handle multiple values
    for value in values:
        if value in similarity_matrix.index:
            # Compute similarity for each species
            species_similarities = character_matrix[character].apply(
                lambda x: max(
                    [similarity_matrix[value].get(v, 0) for v in str(x).split(';') if pd.notna(v)]
                ) if pd.notna(x) else 0
            )
            similarity_scores = np.maximum(similarity_scores, species_similarities)
    
    # Calculate the number of unique non-matching values 
    unique_values = character_matrix[character].astype(str).str.split(';').explode().unique()
    n_unique_values = len(unique_values)
    n_user_values = len(values)
    
    # Handle NaN values in the original character matrix
    nan_mask = character_matrix[character].isna()
    
    # Set similarity scores for NaN values to the adjusted error rate
    if n_unique_values > n_user_values:
        nan_likelihood = error_rate / (n_unique_values - n_user_values)
    else:
        nan_likelihood = error_rate

    similarity_scores[nan_mask] = nan_likelihood
    likelihood *= similarity_scores

    # Handle dummy species
    if 'Not in species list' in character_matrix.index:
        dummy_idx = character_matrix.index.get_loc('Not in species list')
        highest_value = original_matrix.max().max()  # Highest value in the original matrix
        lowest_value = original_matrix.min().min()  # Lowest value in the original matrix
        dummy_likelihood = (highest_value + lowest_value) / 2
        likelihood[dummy_idx] = dummy_likelihood

    total_likelihood = likelihood.sum()
    if total_likelihood > 0:
        likelihood /= total_likelihood
    
    return likelihood


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def clean_country_name(country_string):
    if not isinstance(country_string, str):
        return []
    parts = country_string.split('|')
    cleaned_countries = []
    for part in parts:
        country = part.strip()
        if country:
            cleaned_countries.append(country)
    return cleaned_countries

def compute_species_likelihood(character_matrix, observed_countries, error_rate):
    if isinstance(observed_countries, str):
        observed_countries = clean_country_name(observed_countries)
    else:
        observed_countries = clean_country_name(' | '.join(observed_countries))
    observed_countries = [c for c in observed_countries if c]

    # Build species profiles for each country
    country_profiles = {}
    for species in character_matrix.index:
        countries = clean_country_name(character_matrix.loc[species, 'Distribution'])
        for country in countries:
            if country not in country_profiles:
                country_profiles[country] = set()
            country_profiles[country].add(species)

    # Compute similarity scores between observed countries and all countries in the dataset
    country_similarities = {}
    for observed_country in observed_countries:
        if observed_country not in country_profiles:
            print(f"Observed country '{observed_country}' not found in country_profiles.")
            continue
        country_similarities[observed_country] = {}
        for other_country, other_country_species in country_profiles.items():
            similarity = jaccard_similarity(
                country_profiles[observed_country], other_country_species
            )
            country_similarities[observed_country][other_country] = similarity

    # Compute the likelihood for each species
    num_species = character_matrix.shape[0]
    likelihood_scores = np.zeros(num_species)
    non_matching_indices = []

    for i, species in enumerate(character_matrix.index):
        countries = clean_country_name(character_matrix.loc[species, 'Distribution'])
        if not countries:
            non_matching_indices.append(i)
            continue

        scores = []
        for country in countries:
            for observed_country, similarities in country_similarities.items():
                if country in similarities:
                    score = similarities[country]
                    scores.append(score)
                else:
                    print(f"Country '{country}' not found in similarities for observed country '{observed_country}'.")

        # Calculate average similarity score
        avg_similarity = np.mean(scores) if scores else 0

        observed_countries_set = set(observed_countries)
        species_countries_set = set(countries)
        # Check if any observed country is in the species' countries
        contains_observed_country = any(c in species_countries_set for c in observed_countries_set)
        if contains_observed_country:
            avg_similarity = (avg_similarity - error_rate)
        likelihood_scores[i] = avg_similarity

    # Calculate the number of non-matching states 
    unique_countries = set()
    for countries in character_matrix['Distribution']:
        unique_countries.update(clean_country_name(countries))

    n_unique_countries = len(unique_countries)
    n_user_values = len(observed_countries)
    
    num_non_matching_countries = n_unique_countries - n_user_values

    # Distribute error rate among non-matching states
    if non_matching_indices:
        if num_non_matching_countries > 0:
            non_match_likelihood = error_rate / num_non_matching_countries
        else:
            non_match_likelihood = error_rate  # in case all countries match
        
        likelihood_scores[non_matching_indices] = non_match_likelihood

    # Handle the dummy species likelihood after normalisation but before error adjustments.
    if 'Not in species list' in character_matrix.index:
        dummy_idx = character_matrix.index.get_loc('Not in species list')
        # Compute dummy likelihood as the average of non-normalised likelihood scores
        min_raw_score = np.min(likelihood_scores[likelihood_scores > 0])  # Avoid zero values
        max_raw_score = np.max(likelihood_scores)
        dummy_likelihood = (min_raw_score + max_raw_score) / 2  # Set dummy species likelihood to average score
        likelihood_scores[dummy_idx] = dummy_likelihood

    # Normalise likelihood scores to ensure they sum to 1
    total_likelihood = likelihood_scores.sum()
    if total_likelihood > 0:
        likelihood_scores /= total_likelihood

    return likelihood_scores


def handle_observed_character(character_matrix, character, values, error_rate):
    # Define specific handling based on character
    if character in ['Sex', 'Fovea clearly visible as darkened groove', 'Metatarsus IV dorsally: presence of trichobothrium (TmIV)', 
                     'Tibia IV: number of dorsal spines',
                     'Eyes: appearance',
                     'Anterior cheliceral : appearance',
                     'Maxillae: appearance',
                     'Female palp: claw presence',
                     'Sternum: extends between coxae IV',
                     'Tibia I-II ventrally: presence of spines',
                     'Male pedipalp: femur appearance',
                     'Male pedipalp: paracymbium form',
                     'Male pedipalp: branches of paracymbium presence of teeth',
                     'Male pedipalp: lamella characteristica presence',
                     'Epigyne: form',
                     'Epigyne: seminal receptacles']:
        return binary_character_likelihood(character_matrix, character, values, error_rate)
    elif character in similarity_matrices:
        return categorical_character_likelihood(character_matrix, character, values, error_rate)
    elif character in ['Length of prosoma [mm]', 'Position of trichobothrium on metatarsus I (TmI): relative to metatarsus']:
        likelihood_scores = compute_similarity_scores(character_matrix, {character: values}, error_rate)
        return likelihood_scores
    elif character == 'Distribution':
        # Compute distribution similarity scores
        likelihood_scores = compute_species_likelihood(character_matrix, values, error_rate)
        return likelihood_scores
    else:
        raise ValueError(f"Handling not defined for character: {character}")


def calculate_likelihood(character_matrix, observed_characters, error_rate):
    num_species = character_matrix.shape[0]
    likelihood = np.ones(num_species)
    for character, values in observed_characters.items():
        if character in character_matrix.columns:
            try:
                char_likelihood = handle_observed_character(character_matrix, character, values, error_rate)
                likelihood *= char_likelihood
            except ValueError as ve:
                print(f"ValueError: {ve} (Character: {character}, Values: {values})")
    return likelihood


def calculate_posterior(prior, likelihood):
    with np.errstate(divide='ignore', invalid='ignore'):
        posterior = prior * likelihood
        # Normalise to make it a proper probability distribution
        posterior_sum = posterior.sum()
        if posterior_sum > 0:
            posterior /= posterior_sum
        else:
            posterior = np.zeros_like(posterior)
    return posterior


def rank_species(character_matrix, observed_characters, prior, error_rate):
    likelihood = calculate_likelihood(character_matrix, observed_characters, error_rate)
    posterior = calculate_posterior(prior, likelihood)
    species_posterior = list(zip(character_matrix.index, posterior))
    # Sort by posterior probability
    sorted_species = sorted(species_posterior, key=lambda x: x[1], reverse=True)
    result = sorted_species
    return result

@app.route('/')
def index():
    characteristics = character_matrix.columns.tolist()
    ranked_species = rank_species(character_matrix, {}, global_prior, 0.001)  # Initial ranking without observed characters
    return render_template('app.html', characteristics=characteristics, ranked_species=ranked_species)

@app.route('/get_characters', methods=['GET'])
def get_characters():
    characters = character_matrix.columns.tolist()
    return jsonify(characters)


@app.route('/get_character_values/<character>', methods=['GET'])
def get_character_values(character):
    if character in character_matrix.columns:
        if character == "Length of prosoma [mm]" or character == "Position of trichobothrium on metatarsus I (TmI): relative to metatarsus":
            return jsonify([])  # Return an empty list for these characters
        else:
            unique_values = set()
            for value in character_matrix[character].dropna():
                if isinstance(value, str):
                    unique_values.update([v.strip() for v in re.split(r'[;|]', value)])
                else:
                    unique_values.add(value)
            sorted_values = []
            numerical_values = []
            for value in unique_values:
                if isinstance(value, str) and re.match(r'^\d+(\.\d+)?-\d+(\.\d+)?$', value.strip()):  # Check if it's a numerical range
                    numerical_values.append(value.strip())
                else:
                    sorted_values.append(value)
            numerical_values.sort(key=lambda x: float(x.split('-')[0]))  # Sort numerical ranges by their lower bound
            sorted_values.sort()
            sorted_values.extend(numerical_values)
            return jsonify(sorted_values)
    return jsonify([])

@app.route('/get_initial_species_names', methods=['GET'])
def get_initial_species_names():
    try:
        species_names = character_matrix.index.tolist()
        return jsonify(species_names)
    except Exception as e:
        print(f"Error in get_initial_species_names: {str(e)}")
        return jsonify([])

@app.route('/rank_species', methods=['POST'])
def rank_species_function():
    global global_prior
    try:
        data = request.json  # Parse the JSON data
        observed_characters = data.get('observed_characters', {})
        error_rate = float(data.get('error_rate', 0.001))
        
        for key in observed_characters:
            if not isinstance(observed_characters[key], list):
                observed_characters[key] = [observed_characters[key]]
        ranked_species = rank_species(character_matrix, observed_characters, global_prior, error_rate)
        
        # Update the prior with the latest posterior
        global_prior = calculate_posterior(global_prior, calculate_likelihood(character_matrix, observed_characters, error_rate))
        result = []
        for species, prob in ranked_species:
            result_item = {
                'species': species,
                'probability': prob}
            result.append(result_item)
        return jsonify(result)
    except Exception as e:
        print(f"Error in rank_species_function: {str(e)}")
        return jsonify([])


def kullback_leibler_divergence(p, q, epsilon=1e-10):
    p = np.array(p)
    q = np.array(q)
    p = np.clip(p, epsilon, 1.0) 
    q = np.clip(q, epsilon, 1.0)  
    return np.sum(p * np.log(p / q))

@app.route('/rank_characteristics', methods=['POST'])
def rank_characteristics_function():
    data = request.get_json()
    observed_characters = data.get('observed_characters', {})
    error_rate_for_ranking = data.get('error_rate', 0.0)

    # Calculate initial likelihood and posterior probabilities
    if not observed_characters:
        initial_posteriors = np.ones(character_matrix.shape[0]) / character_matrix.shape[0]
        prior_probs = initial_posteriors
    else:
        prior_probs = np.ones(character_matrix.shape[0]) / character_matrix.shape[0]
        initial_likelihoods = calculate_likelihood(character_matrix, observed_characters, error_rate_for_ranking)
        initial_posteriors = calculate_posterior(prior_probs, initial_likelihoods)

    usefulness = {}
    for character in character_matrix.columns:
        if character not in observed_characters:
            char_values = character_matrix[character].dropna().unique()  # Unique non-null values of the characteristic
            unique_values = set()
            for value in char_values:  # Extract and split unique values if they are strings separated by ';|'
                if isinstance(value, str):
                    unique_values.update([v.strip() for v in re.split(r'[;|]', value)])
                else:
                    unique_values.add(value)

            expected_posterior_changes = []
            for value in unique_values:
                new_obs = observed_characters.copy()
                new_obs[character] = value
                
                # Recalculate likelihood and posterior with the new observation
                new_likelihoods = calculate_likelihood(character_matrix, new_obs, error_rate_for_ranking)
                new_posteriors = calculate_posterior(prior_probs, new_likelihoods)

                # Calculate Kullback-Leibler divergence
                divergence = kullback_leibler_divergence(initial_posteriors, new_posteriors)
                expected_posterior_changes.append(divergence)
            
            if expected_posterior_changes:
                usefulness[character] = np.mean(expected_posterior_changes)
            else:
                usefulness[character] = 0 

    formatted_usefulness = {character: f"{score:.1e}" for character, score in usefulness.items()}
    ranked_characteristics = sorted(formatted_usefulness.items(), key=lambda x: float(x[1]), reverse=True)
    return jsonify(ranked_characteristics)


@app.route('/get_species_details/<species_name>', methods=['GET'])
def get_species_details(species_name):
    species_name = species_name.strip()
    character_matrix.index = character_matrix.index.str.strip()
    species_name_lower = species_name
    matched_species = character_matrix.index[character_matrix.index == species_name][0]
    species_row = character_matrix.loc[matched_species]
    species_details = species_row.to_dict()
    species_details = {key: value for key, value in species_details.items() if not pd.isna(value)}
    species_details['species'] = matched_species
    # Convert NaN values to None
    return jsonify(species_details)

@app.route('/reset_probabilities', methods=['POST'])
def reset_probabilities():
    try:
        global global_prior
        # Reset global_prior to uniform distribution
        initial_probability = 1 / character_matrix.shape[0]
        global_prior = np.ones(character_matrix.shape[0]) * initial_probability
        return jsonify({'message': 'Probabilities and server state reset successfully.'}), 200
    except Exception as e:
        print(f"Error in reset_probabilities: {str(e)}")
        return jsonify({'message': 'Error resetting server state.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
