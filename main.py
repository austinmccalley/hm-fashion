#!/usr/bin/env python3
from valid import customers_with_transactions, create_new_customers
import pickle
import os
import pandas as pd

TRANSACTION_FILE = "./data/transactions_train.csv"
ARTICLES_FILE = "./data/articles.csv"
CUSTOMERS_FILE = "./data/customers.csv"

# Number of rows in transactions
NUM_TRANSACTIONS = 31000000

# Number of transactions to pull into memory at a time
BATCH_SIZE = int(NUM_TRANSACTIONS * 0.01)


def first_run():
    # Check if customers_with_transactions.pkl exists or
    # if new_customers.csv exists
    if(os.path.isfile('./data/customers_with_transactions.pkl') and
            os.path.isfile('./data/new_customers.csv')):
        return

    # Get the customer ids from the sample submission
    cwt = customers_with_transactions()

    # Save the customers with transactions to a pickle file
    with open('./data/customers_with_transactions.pkl', 'wb') as f:
        pickle.dump(cwt, f)

    cwt = None

    # Create a dataframe with only the customers who made purchases during the last week of train
    # (which are the only ones that affect competition metric). It formats the predictions as
    # strings like sample_submission.csv
    nc = create_new_customers()

    # Save the new customers to a pickle file
    with open('./data/new_customers.pkl', 'wb') as f:
        pickle.dump(nc, f)

    nc = None


if __name__ == "__main__":
    first_run()