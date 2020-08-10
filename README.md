# HeartFailureProbabilityRegressionAndClassification
Estimating Risk of Death from Heart Failure with Machine Learning

**Introduction**

Cars need fuel pumped into engines for the engine to work properly, water-cooled computers need water to flow through tubes to cool them to prevent overheating, broadband internet goes from a central control system through cabling and optical fibre to deliver its signals to homes and businesses. Just like these systems, the heart is needed to pump blood through the veins of the body to insure survival. However, unlike previous examples, we only have one heart working continuously through our entire life. Sometimes, the heart (though combinations of genetics, lifestyle habits, and other factors which are not the topic of discussion here) is not capable of shouldering this immense load. When the heart tends to give out on a chronic basis, this is condition is referred to as Heart Failure. The goal of this project is to try an establish models that would predict the event of death of heart failure based on several indicators.

**Models & Data**

For the purpose of this project, ten simple models will be used to try and evaluate the risk of a deadly event from Heart Failure given predictors. The data used here is from the Heart failure clinical records Data Set which can be found at http://archive.ics.uci.edu/ml/ datasets/Heart+failure+clinical+records with variable descriptions which are also available at https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5. In the dataset, the following information is given; whether the patient had anaemia (where 0 is not anaemic and 1 is anaemic, as boolean operator logic is applied for many other variables in the dataset), high blood pressure (boolean operator logic), diabetes (boolean operator logic). We are also given sex of the patient (where 0 signifies the patient is female, and 1 is male), if the patient was a smoker (boolean operator logic). The following biological indicators were also included Creatinine PhosphoKinase (CPK) in mcg/L, Ejection fraction in percentage, Platelets in kiloplateletes/mL, Serum Creatine in mg/dL, and Serum Sodium in mEq/L. Finally, we are also given the number of days in the follow-up period (time between being admitted for left ventricular systolic dysfunction and either being discharged or dying), and whether the patient had died or not (following Boolean operator logic once again).

**Methodology (Initial Regression)**

Using packages MASS, mgcv, mda, tree, matrixStats, caret, pROC, & ggplot2 in R, we start by splitting our dataset into 10 randomized groups using split() function; the first eight groups become both our training and validation sets (we will use leave-one-group-out as a cross-validation method), while the last two groups become our test set to draw fine conclusions from. To be able to judge model accuracy, we must have some sort of metric to base our models on. For this we create a model type.RMSE empty vector for each different model; we will use it to store the Root Mean Square Error of the model validated on one of the eight training/validation groups when the model is trained on the seven others. Thus, we train our models eight different times, recording RMSE values for each group, then after taking a small break to look over those results, we train our models over the entire training/ validation set, and use our test set to give final conclusions on model efficiency.

Our first model is the most simplistic one: it is a simple linear regression model using lm() function and taking into account all the predictor variables in the dataset.

The second model is a revised version of the simple linear regression model: instead of using all predictors in the dataset, we first run a simple linear regression on the entire dataset, and through the use of information given by summary() and anova() functions on the model, we keep this time only predictors with a minimum of significance; here these predictors are age, ejection fraction, serum creatinine & time.

The third model is a generalized linear model using the glm() function with the binomial family and the logic link function (in essence the method for logistic regression using glms). To insure that we use the best forumla for this type of model, we use the stepAIC() function with direction set to ‘both’ to get the set of predictors leading to the lowest Akaike Information Criterion.

The fourth model is a generalized additive model using the function gam() with splines for non-boolean operator logic variables and just regression for boolean operator logic variables (if the splines were not necessary, the function would still calculate as a straight line anyways), then by using the same variable selection method as with our revised linear model using the summary() function.

The fifth model is a basic linear discriminant model. It uses the lda() function and takes into account all the variables which do not need to be standardized for discriminant analysis (as it was also not necessary for the above variables).

The sixth, seventh, and eight models are respectively quadratic discriminant, mixed discriminant, and flexible discriminant models (using qda(), mda(), fda() functions respectively) following the same predictor variables logic as the previous linear discriminant model.

The ninth and final are models are closely related; the ninth is a an extensive decision tree model built using tree() function, while the tenth is a pruned version of the decision tree model using set.seed(5) and the tree.cv() function set to minimize deviations based on the tree size.

**Results (Initial Regression)**

To keep this report clean and uncluttered, we will not include every numerical result, and model built, instead giving small examples below of code inputs (the simple linear model computations with fifth group as validation, the linear discriminant model with third group as validation and the pruned decision tree with fifth group as validation) :

*SEE TYPED REPORT WITH IMAGES FOR VALUES*

From these cross-validation metrics, we can hypothesized that the Generalized Additive Model seemed like the best model, followed closely by the Pruned Decision Tree, Generalized Linear, Revised Linear, Linear Discriminant, and Flexible Discriminant Models. When applying our training/validation sets to train models, all models improve against the Test set predictions (which is normal, given the increase in the size of the training set), with the Generalized Additive Model performing best in both cross-validation and testing phases (Cross-Validation Average RMSE of 0.06482254 and Test RMSE of 0.04052547); in contrast, the Mixed Discriminant has the second lowest RMSE for the Test Set (0.04302420), but a Cross- Validation Average RMSE of 0.07155041 (second highest) which is why it cannot be recommended above the Generalized Additive Model who performed well in both phases.
Interestingly, the Pruned Decision Tree Model boasts good metrics for both phases, even though most iterations of the Pruned Decision Tree Models only have Time as the pruning variable. Which also leads to an interesting question; when patients are admitted, can we know how long their follow-up period will be ? While we could also try to build models to try to estimate the length of the follow-up period, we would rather try to stay on course with our initial goal: Predicting probability of death from Heart Failure. Thus, to build some purely analytical models, we will have to re-apply the same steps from before, but without taking the Time variable into account (since it is not predictable in advance).

**Methodology (Second Regression)**

Thus, as already done above, we run our models through cross-validation and testing set, but this time, without taking Time as a predictor variables.

The Simple Linear Model takes into account all variables available except the Time variable.

The Revised Linear Model uses only the predictors with high significance from the above Simple Linear Model, which are Age, Ejection Fraction, & Serum Creatinine.

The Generalized Linear Model finds the relevant through predictors through the stepAIC() function, but disregarding the Time predictor variable on every iteration while searching for the set predictor variables with the lowest associated AIC.

The Generalized Additive Model uses only the significant predictors and splines, which are the following predictor variables as splines; Age, Creatinine PhosphoKinase, Ejection Fraction, & Serum Creatinine.

All Discriminant Models follow the same format using all predictor variables except the Time variable.

The Decision Tree Models uses all predictor variables, with the exclusion of the Time variable, and the Pruned Decision Tree is built from the Decision Tree that excludes Time variable.

**Results (Second Regression)**

As expected, with the absence of the Time variables, the error metrics for all our models is larger than on our Initial Regression. However, now, our models are purely predictive; they should be able to assess the probability of someone dying from Heart Failure, following admittance to a Hospital. The most notable change is for the Pruned Decision Tree Model, where there there a more than one branch on every iteration (in our Initial Regression, almost all Pruned Decision Tree Models had only the Time variable as a branch alone or without another variable). We have included the Pruned Decision Tree from our Testing Set phase, below as an example, followed by a table indicating the new value for our error metrics in the Second Regression.

*SEE TYPED REPORT WITH IMAGES FOR VALUES*

We can note an improvement for all models here with the worst being the Original Decision Tree with a 5.99% RMSE. While many of our models are close to its RMSE, the best model we have is our Generalized Additive model, since it had a low error score in our cross- validation as well as in our test phase (others either have a great test score, with unremarkable cross-validation results, or great cross-validation results with a test score in the same range as other models).

However what we get our approximate percentages of dying from Heart Failure according to biological indicators used as predictors (they are approximate because some cases may give us negative probabilities or probabilities above 100% [which could be categorized as impossible to die from Heart Failure or death from Heart Failure inevitable respectively]).

Then, let us test our models this way one more time, but this time, with the goal being classification (death vs no death) instead of computed predicted probability of the death event.

**Methodology (Classification)**

Classification implies that the model classifies the outcome (here either death or survival) depending on the predictor variables the model is trained. However, using RMSE would be insufficient in this case to determine the best model to be used, since we can have false negatives and false positives which are both wrong, but indicate different weaknesses in the model. Thus, in addition to our empty RMSE vector, we also set up empty Accuracy, Precision, Recall, F1 Score, AUC value vectors, as well as empty list for ROC curves, and our Confusion Matrices.

Accuracy is the percentage of correct predictions by the model. Precision is the percentage of correctly predicted positive values over all predicted positive values. Recall is the percentage of correctly predicted positive values over all actual positive values. The F1 Score is an harmonic mean of both Precision and Recall (it is at its highest when both Precision and Recall are high). The AUC score corresponds to the Area Under Curve of the Receiver Operating Characteristic Curve which we also store in a list to plot all together. A perfect model would have an AUC of 1 (but can be as low as 0 for a model that cannot predict anything correctly, where an average model would have an AUC of 0.5). The Confusion Matrices are simple 2x2 matrices showing true positives, false positives, false negatives, and true positives from our model results.

For our Simple Linear, Revised Linear, Generalized Linear, Generalized Additive Models, Classification is done by computing the probabilities as in the Second Regression, and then classifying the outcome by round up or down to the closest to 0 or 1 (ie. 0.4 becomes 0, 0.7 becomes 1, 0.5 becomes 1, 0.25 becomes 0, etc...)

For our Discriminant based Models, by changing the section type in the function predict() for Linear and Quadratic Discriminant from ‘posterior’ to ‘response’ & for Mixed and Flexible Discriminant from ‘posterior’ to ‘class’, then the output of the prediction is then outputted as either 0 or 1 (which we convert into numerical values from factors for some calculations).

For our Tree based Models, the change is simply to input DEATH_EVENT as a factor using the as.factor() function in our tree() function to build a classification model instead.

**Results (Classification)**

See below, the code output for Generalized Additive Model on the Test Set, the Set of 8
ROC curves from the 8 cross-validation results from Generalized Linear Model, as well the
Decision Tree Model, the Cross-Validation Tree Size against Deviation Graph & Pruned
Decision Tree Model on the Test Set, as examples for this section:

*SEE TYPED REPORT WITH IMAGES FOR VALUES*

From these results, we can already identify certain issues with our models; on one part, all models (Tree based models aside) have a decent F1 score (around 80%), but whereas our Tree based models have balanced Precision and Recall averages (~80% as well), the other models have a worse Precision (~high 70%), and a better Recall (~high 80%). This indicates that the models are good at identifying true non-fatal cases amongst the non-fatal cases predicted, but that models also tend to predict non-fatal cases where the actual outcome is fatal. Thus, based on Cross-Validation, the Tree based Models seem to be superior, even if the other modes are not far behind.

Then, on our Test set, the following error metrics are produced by our analysis:

*SEE TYPED REPORT WITH IMAGES FOR VALUES*

On our Test set, however, our Tree based models seem to falter, where some Linear and Discriminant Models improve (most notably Generalized Linear, Linear Discriminant, Mixed Discriminant, & Flexible Discriminant Models). However, given the last of consistency between our results from cross-validation and testing set phases, there are no models that we can recommend in the absolute.

**Conclusions**

Ideally, this dataset would be enlarged in the future with more data being sporadically added (as this might help improve the models). It may also be interesting to attempt any kind of clustering analysis to see if there significant differences between clusters (whether it be predictor values or actual death event outcomes) before using our machine learning models (this may lead to better improved, but would also depend on cluster sizes; where a cluster being too small outweighs model improvement). There is also the possibility of using other types of models used in Machine Learning, as the ones used here share a good amount of similarities with each other; but given the sheer amount of models that would need to be tested, this project only made use of those ten, for simplicity’s sake.

Any comments or questions about this report, the R code may be redirected to dpeslherbe@outlook.com or the contact me section of my personal website @ https://dpeslherbe.wixsite.com/website. All relevant files are also included in this shareable Google Drive Folder @ https://drive.google.com/drive/folders/1ftWU0FNWvibInThAMagt8SU-VN-rkEeT?usp=sharing.
