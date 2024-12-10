from glob import glob
import pandas as pd

def load_subtitles_dataset(dataset_path):
    subtitles_paths = glob(dataset_path + '/*.ass')

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        try:
            # Read Lines with explicit encoding
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines = lines[27:]  # Skip the header lines
                lines = [",".join(line.split(',')[9:]) for line in lines]

            # Clean lines and prepare script
            lines = [line.replace('\\N', ' ') for line in lines]
            script = " ".join(lines)

            # Extract episode number
            episode = int(path.split('-')[-1].split('.')[0].strip())

            scripts.append(script)
            episode_num.append(episode)

        except UnicodeDecodeError:
            print(f"UnicodeDecodeError encountered for file: {path}. Skipping this file.")
            continue  # Skip files with encoding issues

    # Create DataFrame
    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df
