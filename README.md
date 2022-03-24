# H&M Personalized Fashion Recommendations

This is a personal project for the [H&M Personalized Fashion Recommendations Challenge](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) from [Kaggle](https://www.kaggle.com/).

## Competition Description

For this challenge you are given the purchase history of customers across time, along with supporting metadata. Your challenge is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends. Customer who did not make any purchase during that time are excluded from the scoring.

## Files

You are able to download the necessary data files by using the following command

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
```


* `data/images/` - images of products
* `data/articles.csv` - detailed metadata for each `article_id` available in the dataset
* `data/customers.csv` - metadata for each `customer_id` in dataset
* `data/sample_submission.csv` - sample submission file for the competition
* `data/transactions_train.csv` - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the article_ids each customer will purchase during the 7-day period immediately after the training data period.
