#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRANSACTION_FILE = "./data/transactions_train.csv"
ARTICLES_FILE = "./data/articles.csv"
CUSTOMERS_FILE = "./data/customers.csv"

# Lets start by reading in the head of all the files

transaction_head = pd.read_csv(TRANSACTION_FILE, nrows=5)
articles_head = pd.read_csv(ARTICLES_FILE, nrows=5)
customers_head = pd.read_csv(CUSTOMERS_FILE, nrows=5)

print(transaction_head)
print(articles_head)
print(customers_head)


