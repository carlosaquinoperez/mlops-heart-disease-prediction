
def to_dicts(df):
    """
    Helper function to convert DataFrame to dictionaries.
    Needed by the scikit-learn pipeline.
    """
    
    return df.to_dict(orient='records')