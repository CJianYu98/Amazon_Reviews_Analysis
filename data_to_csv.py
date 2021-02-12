import pandas as pd
from pandas.io.json import json_normalize
import json
import os
import glob

# please change this path to your own path when you are running
path = '/Users/chenjianyu/Desktop/Y2S2/IS450 Text Mining and Language Processing/Project/AmazonReviews/'

product_files = [filename for filename in os.listdir(path) if filename != '.DS_Store']

for product in product_files:
    product_df = pd.DataFrame()

    product_file_path = path + product
    for filename in glob.glob(os.path.join(product_file_path, '*.json')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            data = json.load(f) #loading json data file
            df = json_normalize(data) # normalizing json data to flatten the data so that it can be converted to Pandas dataframe
            df1 = pd.DataFrame(df['Reviews'][0]) # extracting the reviews and store in a temp df
            if len(df) != 0: # check if df is empty
                # empty df means that there was no text review by that user, and we want to remove such data as it is not meaningful to us
                df1['Product_ID'] = df.iloc[0]['ProductInfo.ProductID']
                df1['Product_Name'] = df.iloc[0]['ProductInfo.Name']
                df1['Product_Features'] = df.iloc[0]['ProductInfo.Features']
                df1['Product_Price'] = df.iloc[0]['ProductInfo.Price']
                
                product_df = product_df.append(df1, ignore_index = True) # append into the product df created earlier if there is a review
    
    # drop records with no reviews
    product_df.dropna(subset = ['Content', 'Product_Name'], inplace = True)

    # exporting processed data into csv file
    os.mkdir('Cleaned Data') # creating a new directory to store the data
    product_df.to_csv(f'Cleaned Data/{product}_reviews.csv') 