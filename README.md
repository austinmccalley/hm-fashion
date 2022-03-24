# H&M Personalized Fashion Recommendations

This is a personal project for the [H&M Personalized Fashion Recommendations Challenge](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) from [Kaggle](https://www.kaggle.com/).

## Competition Description

> H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Our online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are key. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.
> In this competition, H&M Group invites you to develop product recommendations based on data from previous transactions, as well as from customer and product meta data. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images.
> There are no preconceptions on what information that may be useful – that is for you to find out. If you want to investigate a categorical data type algorithm, or dive into NLP and image processing deep learning, that is up to you. 

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

## Notes

_This is going to be collection of my notes as I go through competing in this challenge._

My first thought is to start small and just classify the users into different categories based off of the articles they have purchased. Using this we can start to get a baseline for predicting things. After messing around with clustering of customers using their attributes there wasn't any clear clusters. So I decided to concatenate their last purchases together and use that as the training data.

My next thought is to feed in a product details and make a psuedouser which would buy the product. This we can then iterate over all the products and get the average user for each item, then we can apply a clustering algorithm with these psuedousers and throw real users into the clusters to see where they fall. 