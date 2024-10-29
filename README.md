# GPTrans
Accurately predicting mutations in G protein-coupled receptors (GPCRs) is critical for advancing disease diagnosis and drug discovery. In response to this imperative, GPTrans has emerged as a highly accurate predictor of disease-related mutations in GPCRs. The core innovation of GPTrans resides in its ability to integrate features from both wildtype and mutant protein variation sites, utilizing multi-feature concatenation within a Transformer framework to ensure comprehensive feature extraction. A key aspect of GPTrans’s effectiveness is its deep feature combination strategy, which merges multiple evolutionary-scale modeling amino acid embeddings and class tokens with ProtTrans embeddings, thus shedding light on the biochemical properties of proteins. Leveraging Transformer components and a self-attention mechanism, GPTrans captures higher-level representations of protein features. Employing both wildtype and mutation site information for feature fusion not only enriches the predictive feature set but also avoids the common issue of overestimation associated with sequence-based predictions. This approach distinguishes GPTrans, enabling it to significantly outperform existing methods. Our evaluations across diverse GPCR data sets, including ClinVar and MutHTP, demonstrate GPTrans’s superior performance, with average AUC values of 0.874 and 0.670 in 10-Fold Cross-validation. Notably, compared to the AlphaMissense method, GPTrans exhibited a remarkable 38.03% improvement in accuracy when predicting disease-associated mutations in the MutHTP data set. 
# GPTrans Prediction Performance Comparison
![image](https://github.com/EduardWang/GPTrans/blob/main/pic/final/upsetc.png)
![image](https://github.com/EduardWang/GPTrans/blob/main/pic/final/upsetm.png)
# Install Dependencies
Python ver. == 3.8  
For others, run the following command:  
```Python
conda install tensorflow-gpu==2.5.0
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
# Run
We provide results of two data sets. Run model/predict.py; To train your own data, use the model/fold_cross_validation.py and model/train_test.py file.

# Contact
If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. E-mail: 221210701119@stu.just.edu.cn

