import pandas as pd

TRANSACTION_FILE = "./data/transactions_train.csv"
ARTICLES_FILE = "./data/articles.csv"
CUSTOMERS_FILE = "./data/customers.csv"
SAMPLE_FILE = './data/sample_submission.csv'

""" 
Create a dataframe with only the customers who made purchases during the last week of train 
(which are the only ones that affect competition metric). It formats the predictions as 
strings like sample_submission.csv
"""
def create_new_customers():
    valid: pd.DataFrame = pd.read_csv(TRANSACTION_FILE)
    valid.t_dat = pd.to_datetime(valid.t_dat)
    valid = valid.loc[valid.t_dat >= pd.to_datetime('2020-09-16')]
    valid = valid.groupby('customer_id').article_id.apply(list).reset_index()
    valid = valid.rename({'article_id': 'prediction'}, axis=1)
    valid['prediction'] = valid.prediction.apply(
        lambda x: ' '.join(['0'+str(k)for k in x]))

    return valid

def customers_with_transactions():
    customers: pd.DataFrame = pd.read_csv(SAMPLE_FILE)

    # Get the customer ids from the sample submission
    customer_ids = customers.customer_id.unique()

    return customer_ids
