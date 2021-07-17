# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - the filepath to csv file containing the messages used in this project
    categories_filepath - the filepath to csv file containing the categories for the messages
    
    
    OUTPUT:
    df - dataframe with the result of merging the two input files 
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='left')
    
    return df


def clean_data(df):
    '''
    INPUT:
    df - dataframe with the result of merging the messages and categories files
    
    
    OUTPUT:
    df - the same dataframe cleaned and with the messages and categories as dummy columns 
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories and remove the last two character of each string
    category_colnames = list(categories.iloc[0].str[:-2])
    
    # rename the columns of 'categories'
    categories.columns = category_colnames
    
    # convert category values to numbers (0 or 1)
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        #limit values above 1
        categories.loc[categories[column]>1, column] = 1
        
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - cleaned dataframe 
    database_filename - the filename of the sqlite database to be created
     
    '''
    
    # create sqlite engine
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # save df into table
    df.to_sql('CategorizedMessages', engine, index=False)  


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()