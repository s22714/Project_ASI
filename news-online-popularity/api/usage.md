# How to use the api?

1. Start the uvicorn process
2. Call `Post` endpoint on `http://127.0.0.1:8000/predict` with raw json as body - e.g. data:
   `
{
   "features": {
   "n_tokens_title": 12.0,
   "n_tokens_content": 219.0,
   "n_unique_tokens": 0.663594466988,
   "n_non_stop_words": 0.999999992308,
   "n_non_stop_unique_tokens": 0.815384609112,
   "num_hrefs": 4.0,
   "num_self_hrefs": 2.0,
   "num_imgs": 1.0,
   "num_videos": 0.0,
   "average_token_length": 4.6803652968,
   "num_keywords": 5.0,
   "data_channel_is_lifestyle": 0.0,
   "data_channel_is_entertainment": 1.0,
   "data_channel_is_bus": 0.0,
   "data_channel_is_socmed": 0.0,
   "data_channel_is_tech": 0.0,
   "data_channel_is_world": 0.0,
   "kw_min_min": 0.0,
   "kw_max_min": 0.0,
   "kw_avg_min": 0.0,
   "kw_min_max": 0.0,
   "kw_max_max": 0.0,
   "kw_avg_max": 0.0,
   "kw_min_avg": 0.0,
   "kw_max_avg": 0.0,
   "kw_avg_avg": 0.0,
   "self_reference_min_shares": 496.0,
   "self_reference_max_shares": 496.0,
   "self_reference_avg_sharess": 496.0,
   "weekday_is_monday": 1.0,
   "weekday_is_tuesday": 0.0,
   "weekday_is_wednesday": 0.0,
   "weekday_is_thursday": 0.0,
   "weekday_is_friday": 0.0,
   "weekday_is_saturday": 0.0,
   "weekday_is_sunday": 0.0,
   "is_weekend": 0.0,
   "LDA_00": 0.500331204081,
   "LDA_01": 0.378278929586,
   "LDA_02": 0.0400046751006,
   "LDA_03": 0.0412626477296,
   "LDA_04": 0.0401225435029,
   "global_subjectivity": 0.521617145481,
   "global_sentiment_polarity": 0.0925619834711,
   "global_rate_positive_words": 0.0456621004566,
   "global_rate_negative_words": 0.013698630137,
   "rate_positive_words": 0.769230769231,
   "rate_negative_words": 0.230769230769,
   "avg_positive_polarity": 0.378636363636,
   "min_positive_polarity": 0.1,
   "max_positive_polarity": 0.7,
   "avg_negative_polarity": -0.35,
   "min_negative_polarity": -0.6,
   "max_negative_polarity": -0.2,
   "title_subjectivity": 0.5,
   "title_sentiment_polarity": -0.1875,
   "abs_title_subjectivity": 0.0,
   }
}
`
3. Enjoy the prediction. For the given data expected value is `593.0`
4. ![img.png](img.png)