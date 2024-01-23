# Beyond-Sentiment-An-algorithmic-strategy-for-identifying-evaluations-within-large-text-corpora
The repository contains the scripts, data files, and additional material that were used in the article “Beyond Sentiment: An algorithmic strategy for identifying evaluations within large text corpora" published in Communication Methods and Measures.

To cite: Overbeck, M., Baden, C., Aharoni, T., Amit-Danhi, E., & Tenenboim Weinblatt, K. (2023). Beyond sentiment: An algorithmic strategy for identifying evaluations within large text corpora. Communication Methods and Measures, 1–22. https://doi.org/10.1080/19312458.2023.2285783

1.	The Supplementary Material.docx file with all the supplementary material (S1-S5) used for this article. 
2.	The SVM_Classification_Script.R script used for the SVM-based evaluations classifier and its comparison against sentiment-dictionary based automated classifiers 
3.	The SVM_Classification_Twitter.R script that uses the same SVM-Classifer but additionally calculates performance specifically for Twitter or Media Data in the test-sets 
4.	The roBERTa_evaluations_classifier.py script used for training and evaluating a roBERTa-based evaluations classifier based on term-level training data 
5.	The roBERTa_evaluations_segment.ipynb script which trains and evaluates the robERTa-based evaluations classifier based on segment-level training data.
6.	The roBERTa_sentiment_classifier.py script that runs an off-the-shelf roBERTa-based sentiment classifier on our term-level dataset
7.	The Classifications_Input.xlsx dataset that contains the term-level annotation used for training and testing the term-level classifier
8.	The Classifications_Output.xlsx dataset that contains all classifications made by the supervised classifier. The document contains three tabs:
a.	The first tab All_Classifications includes the term-level classification output.
b.	The second tab Classifications_Segment_Level contains a modified classification output with classifications aggregated on a segment level. This tab also contains the training data used in roBERTa_evaluations_segment.ipynb to train a segment-level evaluations classifier.
c.	The third tab Gold_Standard_Scores contains the calculated results with all correlations and confusion matrices presented in Table 6 in the article to evaluate the performance of the proposed approach against alternative classifiers. 
10.	The Classifications_Output.xlsx dataset that contains the segment-level classification output and performance evaluations. The third tab Gold_Standard_Scores contains the calculated scores and assessments presented in Table 6.
11.	The Confusion Matrices_SVM_NB_roBERTa.xslx file which contains all confusion matrices generated by 10-fold cross validation for the SVM, Naïve Bayes, and roBERTa-based evaluations classifiers as presented in Table 4 in the article.
12.	The create_confusion_matrices.py script and goldstandard_trinary.txt file used to calculate the confusion matrices as presented in Table 6 of the manuscript.
13.	The elections_filter.txt, future_dictionary.txt., sentiment_dictionary.txt files used for sampling and pre-annotating the dataset in preparation for the classification task (steps 1, 2, and 3 of the classification workflow as depicted in Figure 1 in the article)
