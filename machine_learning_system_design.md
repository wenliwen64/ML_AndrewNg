# Machine Learning System Design

1. Priotirizing what to work on(eg. email spam system)
   - How to spend time to make it have low error
       - Collect lost of data
          - E.g. "honeypot" project
       - Develop sophisticated features based on email routning info. 
       - Email body. (discounts/discount, deal/Dealer, punctuation)
       - Mispelling

2. Error Analysis
   - Implement quickly;
   - Plot learning curves for training and cv data set , more features? more data? 
   - Error analysis: look at the eamil misclassified maunally, to design new features;
       - what type of eamil it is(classify them manually)
       - what cues you think would have helped?(steal password)
       - go through features which is important (appears the most often)
  - The importance of numerical evaluation
       - stemming software to detect discodunt/discounters/distcounted as the same word;
       - without stemming(percentage), with stemming(percentage), error metrics
3. Error Metrics for Skewed Data
   - What a joke!:
       ![](http://i.imgur.com/z9IXsGz.png)
   - skewed classes(one class has way larger capacity)
   - one of evaluation metrics(one possible solution)===>Precision/Recall:
   
   |act/pre |1|0|
   |---|---|---|
   1|True positive|False positive|
   0|Flase negative|True negative|
   
   Definition: $$Precision(rare cases) = \frac{\# true\_positives}{\# predicted\_positives}$$; $$Recall(rare cases) = \frac{\# true\_positives}{\# actual\_positives}$$
   High precision / recall means good classifier.
   if we cheat using predict all of the patient have no cancer, then the recall will be high.
   - Tradeoff precision and recall(how to control):
       - improve hypothesis output threshold(0.5->0.9): improve precision, decrease the recall
       - lower threshold,: avoid false negatives
       - like ROC/ plot precision vs. recall;
       - how to decide->F_1 score: $$F_1=2\frac{PR}{P+R}$$ better test on cv dataset to choose the right one
4. Data for machine learning：
   - Large data rationale: 1. use learning algo with many features(, manually look at the data if feature is enouth for human expert.low bias)； 2. and use a very large training set(low variance);
