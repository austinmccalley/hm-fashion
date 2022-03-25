#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from cluster import run_cluster
from tqdm import tqdm

TRANSACTION_FILE = "./data/transactions_train.csv"
ARTICLES_FILE = "./data/articles.csv"
CUSTOMERS_FILE = "./data/customers.csv"

# Number of rows in transactions
NUM_TRANSACTIONS = 31000000

# Number of transactions to pull into memory at a time
BATCH_SIZE = int(NUM_TRANSACTIONS * 0.01)

# Lets start by reading in the head of all the files
def get_heads(verbose=False):
  transaction_head = pd.read_csv(TRANSACTION_FILE, nrows=5)
  articles_head = pd.read_csv(ARTICLES_FILE, nrows=5)
  customers_head = pd.read_csv(CUSTOMERS_FILE, nrows=5)

  if verbose:
    print(transaction_head)
    print(articles_head)
    print(customers_head)

  return transaction_head, articles_head, customers_head


def get_transactions(nrows=None):
  transactions = pd.read_csv(TRANSACTION_FILE, nrows=nrows)
  return transactions

def get_articles(nrows=None):
  articles = pd.read_csv(ARTICLES_FILE, nrows=nrows)
  return articles

def get_customers(nrows=None):
  customers = pd.read_csv(CUSTOMERS_FILE, nrows=nrows)
  return customers

def find_transaction(transaction_id):
  current_page = 0

  while (current_page + 1) * BATCH_SIZE < NUM_TRANSACTIONS:
    transactions = pd.read_csv(TRANSACTION_FILE, nrows=BATCH_SIZE, skiprows=current_page * BATCH_SIZE)
    current_page += 1

    if transactions['transaction_id'] == transaction_id:
      return transactions

  return None

def has_transactions(customer_id: str, customers_with_transactions: set):
  # Check if customer_id is in the set
  cwt = list(customers_with_transactions)
  return customer_id in cwt

def find_transactions_by_customer(customer_id: str):
  current_page = 0

  while (current_page + 1) * BATCH_SIZE < NUM_TRANSACTIONS:
    transactions = pd.read_csv(TRANSACTION_FILE, nrows=BATCH_SIZE, skiprows=current_page * BATCH_SIZE)
    current_page += 1
    print(current_page)

    # Drop all rows which have no value for customer_id
    transactions = transactions.loc[transactions['customer_id'] == customer_id]

    return get_article_id(transactions.loc[transactions['customer_id'] == customer_id])

  return None

def get_unique_customers_transactions():
  customer_ids = []
  current_page = 0

  pbar = tqdm(total = 100)
  while(current_page + 1) * BATCH_SIZE < NUM_TRANSACTIONS:
    transactions = pd.read_csv(TRANSACTION_FILE, nrows=BATCH_SIZE, skiprows=current_page * BATCH_SIZE)
    current_page += 1
    pbar.update(1)

    customer_ids.extend(transactions['customer_id'].unique())
    customer_ids = list(set(customer_ids))

  return set(customer_ids)    

def get_article_id(transactions: pd.DataFrame):
  return transactions.iloc[0]['article_id']

def save_df_csv(df: pd.DataFrame, filename: str):
  df.to_csv(filename, index=False)