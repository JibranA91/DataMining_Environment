

def data_summary(dataframe):
    import pandas as pd

    df_summary = pd.DataFrame(dataframe.describe().transpose())
    df_summary = df_summary[['unique', 'top', 'freq']]
    df_summary.rename(columns={'unique': 'Unique_Values', 'top': 'Top_Value', 'freq': 'Top_Value_Counts'}, inplace=True)

    null_counts = pd.DataFrame(dataframe.isnull().sum(), columns=['Null_Counts'])
    merged_data = df_summary.merge(null_counts, left_index=True, right_index=True)

    merged_data['Column'] = merged_data.index
    merged_data = merged_data[['Column', 'Unique_Values', 'Null_Counts', 'Top_Value', 'Top_Value_Counts']].sort_values(by=['Unique_Values'], ascending=False)

    return merged_data
