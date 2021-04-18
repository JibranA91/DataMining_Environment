def pull_data(csv_path):
    import pandas as pd
    song_attributes = pd.read_csv(csv_path)
    return song_attributes