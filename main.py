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


if __name__ == "__main__":
  pass