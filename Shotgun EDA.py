import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def shotgun_eda(df, target_col):
    '''
    df: Dataframe
    target_col: Name of the column being targetted in the dataframe
    '''
    
    #Print out info about the numerical data
    numeric_data = df._get_numeric_data()
    print('--Numeric--')
    print(numeric_data.info(verbose=True))
    print(numeric_data.describe())
    
    #Print out categorical data
    categorical_data = df.select_dtypes(exclude=["number"])
    print('\n\n--Categorical--')
    print(categorical_data.info(verbose=True))
    print(categorical_data.describe())
    
    #Correlate features
    if target_col != None:
        corr_target = abs(df.corr()[target_col])
        corr_features = corr_target[corr_target>0.5]
        print('\n\n --Features correlated over 0.5--')
        print(corr_features)

    #Visualize data
    try:
        numeric_data.plot.box(figsize=(20, 20), subplots=True)
        plt.show()
        df.hist(figsize=(20, 20), bins=20, grid=False)
        plt.show()
        df.plot.kde(figsize=(20, 20), subplots=True)
        plt.show()
    except:
        print('Error creating plots')
    
    #Create a pairplot and heatmap if the features are small enough
    if len(df.columns) < 20:
        sns.pairplot(df)
        plt.show()
        sns.heatmap(df.corr(), annot=True)
        plt.show()

def main():
    data = pd.read_csv('test.csv')
    shotgun_eda(data, 'Fare')

if __name__ == '__main__':
    main()
