
## THIS ISNT BEING USED. CAN BE DELETED UNLESS WANT TO BE USED LATER

def select_columns(new_column_names, selected_columns, data):
    
    '''
    This function is used to create a new Dataframe by filtering selected columns
    then returning the new dataframe.

    Args:
        new_column_names : list
        selected_columns : list
        data : pandas.dataframe
    
    Returns:
        pandas:DataFrame
    '''

    dataframe = data[selected_columns]
    dataframe.columns = new_column_names

    return dataframe