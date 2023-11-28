import pandas as pd
import numpy as np
import ast

def add_and_multiply_vectors(vector_a, vector_b, random_factor):
    return [a * random_factor + b * (1 - random_factor) for a, b in zip(vector_a, vector_b)]

def calculate_and_export_avg_vectors(labels_file = 'img/train/labels.csv', vectors_file = 'img/train/vectors.csv'):
    def load_and_merge_datasets(labels_file, vectors_file):
        # Load the datasets
        labels_df = pd.read_csv(labels_file)
        vectors_df = pd.read_csv(vectors_file)

        # Merge the datasets on the 'name' column
        merged_df = pd.merge(labels_df, vectors_df, on='name')
        return merged_df

    def string_to_list(vector_string):
        # Convert string representation of list to actual list
        return ast.literal_eval(vector_string)

    def calculate_average_vectors(merged_df):
        # Convert vector strings to lists
        merged_df['z_values'] = merged_df['z_values'].apply(string_to_list)

        # Group by 'label' and calculate mean
        average_vectors = merged_df.groupby('label')['z_values'].apply(lambda x: np.mean(np.vstack(x), axis=0))

        return average_vectors

    def export_vectors(average_vectors, output_file):
        # Format the vectors as comma-separated strings enclosed in brackets
        formatted_vectors = average_vectors.apply(lambda x: '[' + ','.join(map(str, x)) + ']')

        # Export the formatted vectors to a CSV file
        formatted_vectors.to_csv(output_file, header=True)


    merged_df = load_and_merge_datasets(labels_file, vectors_file)
    average_vectors = calculate_average_vectors(merged_df)
    export_vectors(average_vectors, 'average_vectors.csv')

#calculate_and_export_avg_vectors()