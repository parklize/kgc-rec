# Recommending Knowledge Concepts on MOOC Platforms with Meta-path-based Representation Learning

This repository contains the implementation for [EDM2021](https://educationaldatamining.org/edm2021/) paper - "Recommending Knowledge Concepts on MOOC Platforms with Meta-path-based Representation Learning". This work is inspired by and built on top of [ACKRec](https://github.com/JockWang/ACKRec). 



## Abstract

Massive Open Online Courses (MOOCs) which enable large- scale open online learning for massive users have been playing an important role in modern education for both students as well as professionals. To keep users' interest in MOOCs, recommender systems have been studied and deployed to recommend courses or videos that a user might be interested in. However, recommending courses and videos which usually cover a wide range of knowledge concepts does not consider user interests or learning needs regarding some specific concepts. This paper focuses on the task of recommending knowledge concepts of interest to users, which is challenging due to the sparsity of user-concept interactions given a large number of concepts. In this paper, we propose an approach by modeling information on MOOC platforms (e.g., teacher, video, course, and school) as a Heterogeneous Information Network (HIN) to learn user and concept representations using Graph Convolutional Networks based on user-user and concept-concept relationships via meta-paths in the HIN. We incorporate those learned user and concept representations into an extended matrix factorization frame- work to predict the preference of concepts for each user. Our experiments on a real-world MOOC dataset show that the proposed approach outperforms several baselines and state- of-the-art methods for predicting and recommending concepts of interest to users.



## Main environments

**Laptop used for experiments**: Intel(R) Core(TM) i5-8365U processor laptop with 16GB RAM

**Main packages:** Python 3.6; Tensorflow 1.13.1



## Folder structure

```python
├── data          # the folder contains MOOCCube data (input) used for experiments
├── output        # output folder 
requirements.txt  # packages to be installed using pip install -r requirements.txt
data_utils.py     # for evaluation of predicted results using trained model
data_utils.ipynb  # for data preprocessing
m_train.py        # for training the model
m_inits.py
m_layers.py
m_models.py
m_utils.py
metrics.py
```



## Usage

Use the following command to train our method $MOOCIR_{a1}$ on the MOOCCube dataset. The output include ```m_rating_pred_bestmrr.p``` file for predicted item score matrix for all users.

```bash
$ python m_train.py
```

After the above step, you can use ```data_utils.py``` to get the results regarding evaluation metrics on the test set. The default setting ```m_rating_pred_bestmrr.p```  from above for the variable  ```pred_matrix_f```   in the  ```data_utils.py```.

```bash
$ python data_utils.py
```



## Citation

Guangyuan Piao, "Recommending Knowledge Concepts on MOOC Platforms with Meta-path-based Representation Learning", Educational Data Mining, Paris, France, 2021. [[PDF](https://parklize.github.io/publications/EDM2021.pdf)] [[BibTex](https://parklize.github.io/bib/EDM2021.bib)]

