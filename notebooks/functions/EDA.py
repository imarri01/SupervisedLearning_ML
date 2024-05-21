def load_json_files(json_directory):
    json_files = glob.glob(os.path.join(json_directory, '*.json'))
    dataframes = [pd.read_json(file) for file in json_files]
    return dataframes


def merge_dataframes(dataframes):
    return pd.concat(dataframes, ignore_index=True)


def clean_dataframe(df):
    df = df.dropna()
    if 'details' in df.columns:
        details_df = json_normalize(df['details'])
        df = df.drop(columns=['details']).join(details_df)
    return df


def perform_eda(df):
    print(df.describe())
    sns.heatmap(df.corr(), annot=True)
    plt.show()
    sns.pairplot(df)
    plt.show()
    df.hist(bins=30, figsize=(15, 10))
    plt.show()


# Main workflow
json_directory = 'path_to_your_json_files'
dataframes = load_json_files(json_directory)
combined_df = merge_dataframes(dataframes)
cleaned_df = clean_dataframe(combined_df)
perform_eda(cleaned_df)
