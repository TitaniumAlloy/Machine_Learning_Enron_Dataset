
**Machine Learning Project – Identifying Fraud from Enron Emails**

Author: Tanbir

Status: Complete

Completion Date: January 2017

**1.** **Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: &quot;data exploration&quot;, &quot;outlier investigation&quot;]**

The goal of this project is to use machine learning to identify people of interest in the Enron scandal by looking over their employment history. What makes machine learning really useful for this is because it saves time and money by having an automated system to identify the person of interest. If humans were to do this purely then it would definitely take longer.

From the initial run of gathering the dataset we can see that there are a total of 146 employees which includes 18 person of interest. The rest 128 are non-person of interest.

There are a total of 21 features within the dataset. Each one is categorized in to three categories. One is financial feature which consist of salary, deferral\_payments, total\_payments, loan\_advances, bonus, restricted\_stock\_deferred, deferred\_income, total\_stock\_value, expenses, exercised\_stock\_options, long\_term\_incentive, restricted\_stock, director\_fees, and other.

The other category is email feature which consist of to\_messages, email\_address, from\_poi\_to\_this\_person, from\_messages, from\_this\_person\_to\_poi, and shared\_receipt\_with\_poi.

Final category is poi label which has the poi feature.

After seeing some of the data and using some features I plotted in scatter plots using matplotlib I saw some exponentially high value in each plots. There was a value always on the top right and based on the pdf file that had the list of keys, it seemed like the TOTAL value, which is the summation of every person in the data, is being displayed. So first step, I removed this outlier from the dictionary. I decided to stick with scatter plots due to the fact histogram didn&#39;t show NaN values. When looking at the financial and email feature, &quot;LOCKHART EUGENE E&quot; has NaN on everything when you look at the data. NaN data for all values is useless so I removed him. There was also another value called &quot;THE TRAVEL AGENCY IN THE PARK&quot; which isn&#39;t a name so I removed that as well.

**2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: &quot;create new features&quot;, &quot;intelligently select features&quot;, &quot;properly scale features&quot;]**

One of the features that I made is percentage of receipt share with poi. If the person has a higher percentage of receipt then most likely he can be a poi and also a person heavily involved in the scandal. I also made a bonus\_to\_salary\_ratio because this will show the gap in money between the employers. Finally I made two additional features involving the ratio of mailing from and to between the poi and non-poi. This should also play a similar role as to the percentage of receipt. However the receipt targets people who also hasn&#39;t emailed or perhaps were quiet but involved secretively.

Afterwards I decided to see if these new features that I made would be considered useful with the SelectKBest algorithm. So I decided to use the algorithm to find the score for the top 7 features. My decision for the top 7 came based on the K score and surprisingly the top 7 were the following:

(&#39;exercised\_stock\_options&#39;, 24.815079733218194),

(&#39;total\_stock\_value&#39;, 24.182898678566879),

(&#39;bonus&#39;, 20.792252047181535),

 (&#39;salary&#39;, 18.289684043404513),

(&#39;to\_poi\_ratio&#39;, 16.409712548035792),

 (&#39;deferred\_income&#39;, 11.458476579280369),

(&#39;bonus\_salary\_ratio&#39;, 10.783584708160824)

Two of the features I made were top 7 but the other two weren&#39;t. from\_poi\_ratio and shared\_receipt\_percent features were the other two that were not on top. However, the interesting thing was the K-score for shared\_receipt\_percent was the same as shared\_receipt\_with\_poi K score meaning that by the number of receipt that algorithms can determine it either way without a comparison to the total number of receipt. I ended up checking some of the accuracy score using Naives Bayes first. So I compared the accuracy for full feature of 25 and accuracy with the top 7 K score. Obviously it showed that the top 7 had a slight lower accuracy while the all the features combined had a higher one:

Accuracy of entire 25 features: 0.883720930233

Accuracy of top 7 K-score features: 0.880952380952

After also tampering with the number of K-score I decided to pick 7 and add two of the features I created which didn&#39;t change much in the accuracy in Naïves Bayes and will come in handy for choosing my algorithm. For scaling, Naïves Bayes does not naturally need scaling because the algorithm behind it already has inbuilt scaling by design. Decision Tree also somewhat does not require scaling because the algorithm involves measuring the distant between the points and does it by threshold value which is not affected by the features. However I did scaling for it just to be sure and results were the same. For the other two algorithms, scaling and tuning had to be done because both measures by distant between each points and if any features were to be big in large scale it will dominate and can over shadow the small features.

**3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: &quot;pick an algorithm&quot;]**

I ended up using Naives Bayes due to the fact it showed a good accuracy in my earlier checks. Then I decided to try SVM, Decision Tree, and Kneighbors algorithm. There were slight differences among the accuracy for each but the most significant metric that differed between each drastically were precision and recall. Naïves bayes also had a higher accuracy, precision and recall than most of them.

**4. What does it mean to tune the parameters of an algorithm, and what can happen if you don&#39;t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: &quot;discuss parameter tuning&quot;, &quot;tune the algorithm&quot;]**

Tuning algorithm basically means adjusting the parameters until you get the best performance from that algorithm. You can do it manually by adjusting certain parameters based on the documentation for each algorithm or you can do it automatically using other algorithms such as GridSearchCV. You can choose to not to tune your algorithm but in order to get the best results tuning maybe required. It would be a job of a person to engineer and tune to test out the best results or else the predictions will not work well on other data. For this project I utilized GridSearchCV for tuning my parameters. It tries every combination of parameters and then shows the result for best combination. This also saves lots of time especially for SVM algorithm because SVM algorithm can take significantly long especially when using different parameters. I tried for each classifier algorithm and tuned it based on the following:

- Naïves Bayes algorithm is good as the way it is since it&#39;s designed for best performance automatically.
- For Decision Tree, it was the same as Naives Bayes as in the default parameters are the best.
- SVM the best parameters are: {&#39;_C_&#39;_: 1.0 __,_ &#39;_kernel_&#39;_:_ &#39;_linear_&#39;_,_ &#39;_degree_&#39;_: 3__ ,_ _gamma:_ &#39;_auto_&#39;_,_ &#39;_coef0_&#39;_: 0.0 __,_ &#39;_shrinking_&#39;_: True__ ,_ &#39;_probability_&#39;_: False __,_&#39;_tol_&#39;_: 0.001__ ,_ &#39;_cache\_size_&#39;_: 200 __,_ &#39;_class\_weight_&#39;_: None__ ,_ &#39;_verbose_&#39;_: False __,_ &#39;_max\_iter_&#39;_: -1__ ,_ &#39;_decision\_function\_shape_&#39;_:_ &#39;_vr_&#39;_,_ &#39;_random\_state_&#39;: _None}_
- Finally for Kneighbors the best parameters are:  {&#39;n\_neighbors&#39;: 4, &#39;n\_jobs&#39;: 1, &#39;algorithm&#39;: &#39;auto&#39;, &#39;metric&#39;: &#39;minkowski&#39;, &#39;metric\_params&#39;: None, &#39;p&#39;: 2, &#39;weights&#39;: &#39;uniform&#39;, &#39;leaf\_size&#39;: 1}

**5. What is validation, and what&#39;s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: &quot;discuss validation&quot;, &quot;validation strategy&quot;]**

Validation is to test your machine learning algorithm&#39;s performance on independent data in order to see how well it was trained. A mistake one can do on validation is to test the algorithm on the same data that it was trained on.  The way I dealt with this naturally is make test size split only by 30 percent in the code give as shown: &quot;train\_test\_split(features, labels, test\_size=0.3, random\_state=42)&quot; while the training was 70 percent. This is within the evaluate\_clf function In addition in the function the iteration is set to 100 times so you&#39;ll have the data test 100 times with cross validation technique by splitting the precision, recall and accuracy then grabbing the mean of each metric. This also ensures the numbers when testing on independent data.

**6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm&#39;s performance. [relevant rubric item: &quot;usage of evaluation metrics&quot;]**

The two evaluations metric and average for Naives Bayes are the following:

Accuracy: 0.86050       Precision: 0.51572      Recall: 0.38550

Accuracy is 0.86 meaning it can classify POI vs Non-Poi at 0.86 correctly. Precision on the other hand is how accurate can the algorithm identify POI when an actual POI is given. In this case it&#39;s highly important we get a high value for this in all cases. As Sebastian has stated in precision and recall lesson that getting a high value of precision is important and having 0.51572 means that 51 percent is classified as actual true POIs.  Recall refers how many were classified as POIs from the dataset and it was 0.38 from the dataset.

