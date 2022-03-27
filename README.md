# H&M Personalized Fashion Recommendations

This is a personal project for the [H&M Personalized Fashion Recommendations Challenge](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) from [Kaggle](https://www.kaggle.com/).

## Competition Description

> H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Our online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they might not make a purchase. To enhance the shopping experience, product recommendations are key. More importantly, helping customers make the right choices also has a positive implications for sustainability, as it reduces returns, and thereby minimizes emissions from transportation.
> In this competition, H&M Group invites you to develop product recommendations based on data from previous transactions, as well as from customer and product meta data. The available meta data spans from simple data, such as garment type and customer age, to text data from product descriptions, to image data from garment images.
> There are no preconceptions on what information that may be useful – that is for you to find out. If you want to investigate a categorical data type algorithm, or dive into NLP and image processing deep learning, that is up to you. 

For this challenge you are given the purchase history of customers across time, along with supporting metadata. Your challenge is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends. Customer who did not make any purchase during that time are excluded from the scoring.

## Evaluation

Submissions are evaluated according to the Mean Average Precision @ 12 (MAP@12):

$MAP@12 = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{min(m,12)}  \sum_{k=1}^{min(n,12)} P(k) \times rel(k)$

where $U$ is the number of customers, $P(k)$ is the precision at cutoff $k$,$n$ is the number predictions per customer, $m$ is the number of ground truth values per customer. and $rel(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label and 0 otherwise.

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

My next thought is to pull an `article_id` from the transactions and get all the customers who have purchased that article. Then we can use the customers and the next purchase they will make to predict the article they will purchase.

### Pivot

I took a day off of this and started to think about how we can get recommendations based off a non yes/no purchase history. For example, we dont know what articles a user has looked at previously before and said no to. We only have the "yes's". I read a couple of articles about recommender systems using autoencoders and I think this will be a great starting point for the project. A paper by Florian Strub, Romaric Gaudel and Jérémie Mary outline the process well in their paper [Hybrid Recommender Systems based on Autoencoders](https://dl.acm.org/doi/pdf/10.1145/2988450.2988456).
