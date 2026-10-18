import os
import numpy as np
import json
from collections import Counter

# This is the step before split_dataset.py. 
## The following counts the N and AFIB segments in each patient's .npz files and writes the counts to the label_counts.json. 
def analyze_npz_labels(folder_path, output_folder):

    os.makedirs(output_folder, exist_ok=True) # Sets the output directory

    # Empty Dictionary
    results = {}

    count_sum = Counter()

    # Iterate through the .npz files
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load the .npz file data
                data = np.load(file_path, allow_pickle=True)
                print(f'{filename}:{data['segments'].shape}')

                # Check the labels in the file
                if 'labels' in data:
                    labels = data['labels']

                    # Vectorise the labels
                    labels = np.array(labels).flatten()

                    # Count the number of N and AFIB labels
                    count_N = int(np.sum(labels == 'N'))
                    count_AFIB = int(np.sum(labels == 'AFIB'))

                    # Add to the results dictionary
                    results[os.path.splitext(filename)[0]] = {
                        'N': count_N,
                        'AFIB': count_AFIB
                    }

                    # Update the counter
                    count_sum.update({
                        'N': count_N,
                        'AFIB': count_AFIB
                    })
                else: # If no labels exist, print the error
                    print(f"Error: {filename} has no 'labels'")
                    results[filename] = {'error': 'no labels key'}

            except Exception as e:
                print(f"Error for {filename}: {e}")
                results[filename] = {'error': str(e)}

    # Create the JSON
    print(f'Label Counts：{count_sum}')
    output_file = os.path.join(output_folder, 'label_counts.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False) # Writes the label_counts.json file with the results

    print(f"Label Counts at: {output_file}")


if __name__ == "__main__":
    folder_path = './afdb_dataset/data'
    output_folder = './data_split/bihaf'

    analyze_npz_labels(folder_path, output_folder)