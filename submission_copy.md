## Executive Summary: 

The proposed project aims to address the problem of efficiently categorizing large volumes of computer science research papers using natural language processing (NLP) and machine learning (ML) techniques.The increasing volume of research papers in the field of computer science makes it challenging for researchers and industry practitioners to keep up with the latest advancements and discoveries.We got the data from arxiv "https://arxiv.org/abs/2206.04615", did scrapped the data from Kaggle website https://www.kaggle.com/datasets/Cornell-University/arxiv and then converted it into a CSV file to perform operations on it. Automated categorization can help to streamline this process and make it easier to find and access relevant information.The project involved the use of a dataset containing over 1.7 million research papers from the arXiv repository. The dataset was pre-processed, cleaned, and transformed to prepare it for analysis. Data reduction techniques were used to select only the most relevant data for modeling. Exploratory data analysis (EDA) was conducted to gain insights into the research papers' characteristics and trends in the field of computer science. 

The findings revealed important patterns, such as the dominance of machine learning research, the growth of research papers contributed each year, and the most common words used in abstracts and titles.Based on the EDA findings, we will build 5 models using NLP techniques using Feature Selection techniques such as TF-IDF vs Word2Vec vs Bag of words to automatically categorize research papers into different classes. The various classes were clubbed whenever the relative frequency was less than 3 % for any particular class. Additionally, a couple of Transfer-learning models were built : Vader and BERT to extract emotions out of the data, as there are no sentiment classes already associated with our existing data. 

The model's performance was evaluated using several metrics, and the results showed that it was highly accurate in predicting the categories of research papers. The project's outcomes and recommendations include the development of an automated categorization model that can help researchers and industry practitioners save time and effort in categorizing research papers. The model can also help in identifying research gaps and trends in the field, leading to new discoveries and innovations. The project's benefits extend to researchers, industry practitioners, policymakers, and other stakeholders in the field of computer science. They can benefit from the model's accurate and efficient categorization of research papers, leading to faster access to relevant information and more informed decision-making. The potential problem or opportunity is the increasing volume of research papers in the field of computer science, which makes it challenging to keep up with the latest advancements and discoveries. The proposed solution addresses this problem by providing an automated categorization model that can help streamline the process and make it easier to find and access relevant information. This project is important as it addresses a critical need in the field of computer science and can have significant implications for research, innovation, and technological advancements. As the field continues to grow and evolve, automated categorization models can help keep pace with the increasing volume of research papers and ensure that researchers and industry practitioners have access to the latest information and discoveries.

## Statement of Scope
### Project objectives:
* To pre-process the dataset of research papers by employing NLP techniques such as tokenization, stemming, and stop-word removal, to prepare the text data for modeling.
* To develop a classification model using ML algorithms that can accurately categorize research papers into one of the predefined categories listed in the arXiv taxonomy for computer science.
* To evaluate the efficiency and accuracy of the developed system using various performance metrics, such as precision, recall, and F1 score.

### Variables: 
The project will use the following variables:
* Independent Variable: The research papers dataset from the arXiv repository, consisting of over 1.7 million research papers from various academic disciplines.
* Dependent Variable: The automated system for categorizing research papers developed using NLP and ML techniques.
* Control Variables: The categories of research papers from the arXiv taxonomy for computer science and the performance metrics used for evaluating the developed system.
* Unit of Analysis: The individual research papers from the dataset will be categorized using the developed system.
 

The following categories have been identified in Computer science department from [arxiv taxonomy](https://arxiv.org/category_taxonomy)
| Abbreviation | Meaning |
|:---|:---|
| cs.AI  | Artificial Intelligence
| cs.AR  | Hardware Architecture
| cs.CC  | Computational Complexity
| cs.CE  | Computational Engineering, Finance, and Science
| cs.CG  | Computational Geometry
| cs.CL  | Computation and Language
| cs.CR  | Cryptography and Security
| cs.CV  | Computer Vision and Pattern Recognition
| cs.CY  | Computers and Society
| cs.DB  | Databases
| cs.DC  | Distributed, Parallel, and Cluster Computing
| cs.DL  | Digital Libraries
| cs.DM  | Discrete Mathematics
| cs.DS  | Data Structures and Algorithms
| cs.ET  | Emerging Technologies
| cs.FL  | Formal Languages and Automata Theory
| cs.GL  | General Literature
| cs.GR  | Graphics
| cs.GT  | Computer Science and Game Theory
| cs.HC  | Human-Computer Interaction
| cs.IR  | Information Retrieval
| cs.IT  | Information Theory
| cs.LG  | Machine Learning
| cs.LO  | Logic in Computer Science
| cs.MA  | Multiagent Systems
| cs.MM  | Multimedia
| cs.MS  | Mathematical Software
| cs.NA  | Numerical Analysis
| cs.NE  | Neural and Evolutionary Computing
| cs.NI  | Networking and Internet Architecture
| cs.OH  | Other Computer Science
| cs.OS  | Operating Systems
| cs.PF  | Performance
| cs.PL  | Programming Languages
| cs.RO  | Robotics
| cs.SC  | Symbolic Computation
| cs.SD  | Sound
| cs.SE  | Software Engineering
| cs.SI  | Social and Information Networks
| cs.SY  | Systems and Control

## Project Schedule


![pic1](assets/chart222.png)










## Data Preparation

In this project, data preparation was carried out in two steps: [prepare_dataframe.py](code/prepare_dataframe.py) and [clean_data.py](code/clean_data.py):

* The first step, [prepare_dataframe.py](code/prepare_dataframe.py), involved reading a large JSON file containing metadata about scientific papers from the arXiv preprint server. The script removed irrelevant columns and converted the data into a panda's data frame.
* The second step, [clean_data.py](code/clean_data.py), involved further processing the data to prepare it for machine learning. The script extracted relevant categories from the data, cleaned the text by removing punctuation, stop words, and other non-essential information, and removed duplicate papers. 
* [arxiv-metadata-oai-snapshot_sample.json](data/arxiv-metadata-oai-snapshot_sample.json) is a sample of the original JSON data used in the project. This file was too large to upload to GitHub, so a smaller sample was used instead.

The output of the data preparation steps includes several CSV files. 
* [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) is a tab-delimited file containing computer science papers, which were extracted from the original JSON data and cleaned using the clean_data.py script.
* [text_data_for_modelling_sample.csv](data/text_data_for_modelling_sample.csv) is another tab-delimited file that contains information about each paper's abstract, title, and categories. This file is specifically designed for use in machine learning models and will be used as training data.
* Finally, [cleaned_data_for_eda_sample.csv](data/cleaned_data_for_eda_sample.csv) is a tab-delimited file that contains papers along with additional information, such as the number of categories the paper belongs to, the number of authors, and the length of the abstract. This file was used in exploratory data analysis (EDA) to better understand the characteristics of the data.

Overall, the above steps were critical in preparing the raw data for machine learning models and exploratory data analysis. By cleaning and formatting the data in a consistent and organized way, the project team was able to move on to the next steps of the project with a solid foundation.

### Data Access

The data was sourced from the ‘arxiv’ Database containing information of over hundreds of thousands of research papers in various fields. Data was available on Kaggle. The sourced arxiv papers were in the form of a large JSON, which was difficult to process. Hence, we used the ‘DASK’ open-source computing framework for extracting the data. DASK offers a high-scale parallel computing environment for handling and processing large data.  

Below is a sample record from our JSON data:

```JSON
{"id":"0704.0001",
"submitter":"Pavel Nadolsky","authors":"C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
"title":"Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies",
"comments":"37 pages, 15 figures; published version","journal-ref":"Phys.Rev.D76:013009,2007",
"doi":"10.1103/PhysRevD.76.013009","report-no":"ANL-HEP-PR-07-12",
"categories":"hep-ph","license":null,
"abstract":"  A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which the calculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced from QCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.\n",
"versions":[{"version":"v1","created":"Mon, 2 Apr 2007 19:18:42 GMT"},
{"version":"v2","created":"Tue, 24 Jul 2007 20:10:27 GMT"}],
"update_date":"2008-11-26",
"authors_parsed":[["Bal\u00e1zs","C.",""],["Berger","E. L.",""],["Nadolsky","P. M.",""],["Yuan","C. -P.",""]]}
```
First, we divide the data into chunks of 256 MB and process each chunk in one pass. The following code was used:

```Python
records = dd.read_json('data/arxiv-metadata-oai-snapshot.json',  blocksize='256MB') # Reading the JSON in chunks of 256 MB
 
for i in range(records.npartitions):
    print('Processing '+ str(i)+'th partition')
    try:
        partition_data = records.get_partition(i).compute()
    ...
```

For each chunk, we start with filtering out the papers which had 'cs' as one of their categories. This was done by using the regex:

```Python
pattern='[\s\b]cs\.[A-z.]'
```
The second step was to only pull the required columns. Columns like 'comments', 'versions', 'journal-ref' etc were dropped. There were 2 columns that had author info, one of them being parsed and another one being the string. We chose the parsed column called 'authors_parsed', which made it easy for us to distinguish between authors.
The data from all the chunks was collected into a tab delimited CSV. Also, some chunks had to be processed separately due to 'id' column, whose format was changed by arxiv in later years. 
The last step was to format the authors from format like ["Berger","E. L.",""] to *E L Berger*. This was done by concatenating components of the list in the right order


### Data Cleaning
Data cleaning is an essential step in text analytics as it helps to improve the quality and accuracy of the data used for analysis. Text data often contains noise, errors, and inconsistencies that can affect the performance of the analysis. The following are some common data cleaning techniques used in text analytics:
Removing stop words: Stop words are common words like "the," "and," and "is" which don’t add lot of value to our analysis. So we remove it from our text to improve the performance of our analysis.
Below is the summary of the data cleaning steps we undertook.


#### Removing duplicated titles: 
The dataset contains duplicated titles which add redundancy to our data and will not help with the performance of our analysis. So, we removed the duplicated titles.
*Code Snippet:*
```Python
duplicated_rows = dataframe[dataframe[column].duplicated()]
dataframe.drop_duplicates(subset=[column], inplace=True)
```


#### Removing Non-CS categories:
For our analysis, we are considering only papers published under the Computer Science field. So, we removed all the categories which are not in CS category.
*Code Snippet:*
```Python
pattern = r'\bcs\.\w+' 
dataframe[column] = dataframe[column].apply(lambda x: re.findall(pattern, x))
```


#### Removing the accents:
There are many scientific words which are in different languages than English, for example accent words (like 𝑘). So, we converted them to English word (like k)
*Code Snippet:*
```Python
dataframe[column] = dataframe[column].apply(lambda x: unidecode.unidecode(x))
```


#### Removing punctuations: 
Punctuation marks, such as commas and periods, are removed to eliminate noise and improve the consistency of the data.
*Code Snippet:*
```Python
dataframe[column] = dataframe[column].str.replace(patternpunc,' ')
```


#### Removing single alphabet words:
We have removed the words which are single alphabets in the data because they are irrelevant and can lead to lower accuracy in our analysis.
*Code Snippet:*
```Python
dataframe[column] = dataframe[column].apply(lambda x: re.sub(pattern='\b[A-z]\b', repl="", string=x))
```


### Data Transformation
Data transformation is the process of converting raw text data into a format that is suitable for analysis. The following are the data transformation techniques used in this analytics project:


#### Removing redundant columns: 
We have removed the redundant columns from our dataset and only kept the columns which are required for our analysis.


#### Changing Data Type: 
Based on the data, we have made required changes in datatype of the variables.
*Code Snippet:*
```Python
def change_datatpyes(dataframe, int_columns=None,  date_columns=None):
    
    if int_columns:
        for column in int_columns:
            dataframe[column] = pd.to_numeric(dataframe[column])
                
    if date_columns:
        for column in date_columns:
            dataframe[column] = pd.to_datetime(dataframe[column])
    
    return data
```


#### Converted text to lowercase: 
We have transformed all the text into lowercase in our dataset. Changing the text to lowercase will improve the analysis as all the text is in same case.
*Code Snippet:*
```Python
    dataframe[column] = dataframe[column].str.lower()
```


### Data Reduction
Data reduction happened in multiple stages. The first time it happened was while importing the data, where we did not consider the columns that would be used for our analysis.
In the second part, we removed stopwords, punctuations and single-alphabet words (like 'x', 'y'), These have been described in their respective sections.
Lastly, from this dataset, we prepare 2 data: one for which we keep these columns for EDA. The other data would be used for modelling in future. These are the data reduction steps we undertook for modelling data:
1. As we now know that a paper can have up to 7 target categories, we do not need those many for modelling. We choose up to 3 categories for each paper. If it has less than 3 categories, then less than 3 would be considered. Anything above 3 categories would be trimmed to first 3.
The following piece of code extracts individual categories out of the list of categories. If 2nd and 3rd are not available, it fills an empty string in the value
  ```Python
  data['primary_category'] = data['categories'].apply(lambda x:x[0]) 
  data['secondary_category'] = [i[1] if len(i)>1 else '' for i in data['categories']]
  data['tertiary_category'] = [i[2] if len(i)>2 else '' for i in data['categories']]    
  ```
2. Only the columns 'abstract' and 'title' were considered for modelling as they contain all the text information. Other columns like *submitter*, *authors* etc were not considered. This dataset is stored as [modelling data](data/text_data_for_modelling_sample.csv). Again, due to size limits on GitHub, we were not able to push the entire dataset.


![image](https://user-images.githubusercontent.com/111655952/236694868-8315ce66-cf7b-4527-92c6-a53a50b10662.png)

There are around 150 words on average in every document. We restricted the word count to 150 in each of the documents. We then took a subset of 20 % of the overall data randomly while still preserving the distribution of the classes.

![image](https://user-images.githubusercontent.com/111655952/236694899-aeba92b7-17fc-4c57-b94a-e5a248da0dbe.png) 

Original distribution

![image](https://user-images.githubusercontent.com/111655952/236694919-3ec8dcb4-933c-4efe-a9be-cb8e87d0b8d4.png) 

Sample distribution

We then grouped the minority classes with less than 3% of the data into a class called “Other”.

#### SVD for Data Decomposition :

* SVD can help reduce the dimensionality of data by decomposing the original matrix into three separate matrices: a left singular matrix, a diagonal matrix, and a right singular matrix.

* The diagonal matrix contains the singular values of the original matrix, which can be thought of as a measure of the importance of each dimension in the data. By retaining only the top-k singular values, where k is a smaller number than the total number of singular values, we can reduce the dimensionality of the data while preserving most of its original information. Here we used the matrix obtained from the “Abstract” column to reduce the data. This is used for modelling to avoid overfitting. 

This stage happened while modelling. We had applied 2 types of feature engineering techniques while working with the text data: Bag-of-words and TF-IDF vectors. These two techniques tend to produce high dimensional and sparse embeddings. For our case, we had more than 70,000 dimensions. Having the model built on this high dimensionality would have multiple issues:
 a. It would not be able to identify the key features that impact the output variable y
 b. It would take long time to train.
 
 Hence, these dimensions were reduced to 50 using truncated SVD (Singular Value Decomposition). Following code was used to do that:
 
 ```python
svd = TruncatedSVD(n_components=50)

# Transforming training, validation and testing dataset
X_train_reduced = svd.fit_transform(X_train)
X_valid_reduced = svd.transform(X_valid)
X_test_reduced = svd.transform(X_test)
 ```


#### Lemmatization:

First, a text lemmatizer was used to extract the core lemma from all the words. Lemmatizer was preferred over stemming as it preserves the actual meaning of the word. The lemmatizer was built on a random sample of the data eliminating all bias, that is discussed above.

![image](https://user-images.githubusercontent.com/111655952/236695072-baa7c80d-09a0-4d34-a302-432d7d28985b.png)

### Data Consolidation
For our use case, data was already in the form of a single JSON file. However, due to limitations on RAM size, the data had to be first divided into chunks and then consolidated after removing the non-relevant data (papers that were not related to CS)
This was utilized using both dask and pandas. 
Firstly, the data was read in chunks of 256 MB, as stated in Data Access section. Then, each chunk was filtered for only the data in which 'cs' was in categories. These filtered chunks would be appended to consolidate in a single file.
```Python
records = dd.read_json('data/arxiv-metadata-oai-snapshot.json',  blocksize='256MB') # Reading the JSON in chunks of 256 MB
    
cs_papers_data = pd.DataFrame()  #empty dataframe

for i in range(records.npartitions): # Iterating through all partitions
    print('Processing '+ str(i)+'th partition')
    try:
        partition_data = records.get_partition(i).compute()

        cs_papers_parition_data = filter_cs_papers(partition_data, pattern='[\s\b]cs\.[A-z.]') # Using regex to pull any cs columns

        cs_papers_parition_data.drop(columns=['comments', 'journal-ref', 'doi', 'license', \
                                     'versions', 'authors', 'report-no'], inplace=True, errors='ignore')
        cs_papers_data = cs_papers_data.append(cs_papers_parition_data) # Appending into the dataframe
```
Some of the partitions were processed separately due to the issues with ID column. The format of 'id' column were changed by arxiv, so there were data type issues occurring while calling *compute()* function. Hence the later chunks got processed separately for which data type was defined explicitly.


### Data Dictionary
The dataset contains several variables that provide information about academic works, such as papers or articles.

| Attribute Name | Description | Data Type | Source | Data | Example |
|:---|:---|:---:|:---|:---|:---:|
|ID |	Represents a unique identifier for each record in the dataset |	Integer |	https://www.kaggle.com/datasets/Cornell-University/arxiv | [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) |	704.0001 |
|Submitter	| Person or entity that submitted the record to the dataset.	| String |	https://www.kaggle.com/datasets/Cornell-University/arxiv	|[computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) | Pavel Nadolsky
| Authors |	The authors of the paper that is described |	String |	 https://www.kaggle.com/datasets/Cornell-University/arxiv | [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) | 	E. L. Berger, P. M. Nadolsky |
|Title  |	Represents the title of the paper |	String |	https://www.kaggle.com/datasets/Cornell-University/arxiv | [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) |	Calculation of prompt diphoton production cross sections |
| Categories	| The categories or topics that the paper is related to |	String |	https://www.kaggle.com/datasets/Cornell-University/arxiv | [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) |	hep-ph |
| Abstract	| Summary of the content of the paper	| String	| https://www.kaggle.com/datasets/Cornell-University/arxiv	| [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) | A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy.|
| Update_date |	The date when the record was last updated.|	Timestamp |	https://www.kaggle.com/datasets/Cornell-University/arxiv | [computer_science_papers_sample.csv](data/computer_science_papers_sample.csv) |	11/26/2008 |

| Attribute | Name Purpose |
|:---|:---|
| ID | A unique identifier for each record in the dataset that can be used to keep track of each paper's information. |
| Submitter | The person or entity that submitted the record to the dataset. This information can be used to track submissions and their sources. |
| Authors | The names of the authors of each paper can be used to identify the contributors to each research paper and their respective expertise.|
| Title | The title of each paper can provide information on the subject matter being researched and can be used to categorize papers into different fields of study.|
|Categories | The categories or topics that each paper is related to can be used to analyze the frequency of research in different fields of study, and to identify patterns in the growth of specific research areas over time. |
| Abstract | The summary of the content of each paper can provide a brief overview of the research being conducted and can be used to identify keywords that are commonly used in different fields of study. |
| Update_date | The date when each record was last updated can be used to track changes in research over time, and to identify trends in the evolution of different research topics. |

 
## Descriptive Statistics and Analysis


Let's start with the descriptive analysis for the dataset. 

### The below table shows us the quantitative measures for the complete dataset:

|                            |     Count     |     Mean          |     Median |     Standard Deviation |     Minimum |     Maximum |     25%   |     75%    |
|----------------------------|---------------|-------------------|------------|------------------------|-------------|-------------|-----------|------------|
|     Number_of_authors      |     252598    |     3.736035      |     3      |     3.210502           |     1       |     445     |     2     |     5      |
|     Number_of_categories   |     252598    |     1.906373      |     2      |     0.75728            |     1       |     7       |     1     |     2      |
|     Title_word_count       |     252598    |     7.874465      |     8      |     2.676336           |     0       |     33      |     6     |     9      |
|     Abstract_word_count    |     252598    |     106.703185    |     105    |     36.787049          |     0       |     345     |     81    |     131    |

* Number of Authors: The "Number_of_authors" variable showcases that, on average, each entry in the dataset is associated with around 3.74 authors. However, there is noticeable variability in the number of authors per entry, with values ranging from 1 to as high as 445. This suggests a wide range of collaboration patterns among the entries, with some having only one author while others involve a large number of contributors.

* Number of Categories: The "Number_of_categories" variable reveals that, on average, each entry is assigned around 1.91 categories. The distribution of categories per entry is relatively concentrated, with a low standard deviation of approximately 0.76. This indicates that the majority of entries have two or fewer categories assigned to them. It's possible that the dataset primarily focuses on a limited number of categories, with some entries falling outside this trend and having more categories associated with them.

* Title Word Count: The "Title_word_count" variable provides insights into the length of the entry titles. On average, the titles consist of approximately 7.87 words. The distribution of word counts exhibits moderate variability, with a standard deviation of around 2.68. The range of title word counts spans from 0 to 33 words. This suggests that while most titles fall within the 6 to 9-word range, there are entries with shorter or longer titles as well, including some with no words at all.

### Plot shows the frequency of word counts in the Abstract for the whole dataset :

![image](https://user-images.githubusercontent.com/111641830/236649707-f393be0d-a890-43dc-8660-49c927094a3f.png)

### Plot shows the frequency of word counts in the Title for the whole dataset :

![image](https://user-images.githubusercontent.com/111641830/236649715-396f1753-428d-47cb-8792-5462f41e1842.png)

### The below table shows us the Top 20 dates with the maximum number of publications:

| Update_date | Number_of_Publications |
|:-----------:|:----------------------:|
|  2007-05-23 | 2656                   |
|  2008-02-03 | 672                    |
|  2023-03-07 | 576                    |
|  2023-01-18 | 561                    |
|  2023-03-14 | 551                    |
|  2023-01-31 | 547                    |
|  2023-02-28 | 537                    |
|  2022-06-22 | 535                    |
|  2022-09-07 | 528                    |
|  2016-11-17 | 501                    |
|  2022-11-29 | 481                    |
|  2023-02-07 | 480                    |
|  2023-02-21 | 477                    |
|  2022-10-18 | 470                    |
|  2022-11-15 | 469                    |
|  2022-11-22 | 454                    |
|  2022-10-25 | 451                    |
|  2023-02-14 | 436                    |
|  2022-01-19 | 430                    |
|  2022-12-06 | 422                    |


### The bar chart shows the number of publications over the years :

We see a clear trend, the number of publications have increased over the years. In 2002, less than 5,000 papers were published and in the year 2022, more than 70,000 papers were published. 

![image](https://user-images.githubusercontent.com/111641830/236649725-1215405b-39a0-481b-9b5f-4abbc54c18d3.png)

### The bar chart shows the number of publications based on the months :

We do not see any specific trend in the number of publication based on the months. Almost all the months have around similar number of papers published.

![image](https://user-images.githubusercontent.com/111641830/236649941-cc0f56ac-8d83-45a5-aed5-efa890ff2f8e.png)

### The bar chart shows the number of publications based on the Week Days :

This is an interesting chart, as we see that most of the papers are published on the weekdays which makes sense as publishing centeres and universities are not working over the weekends (Saturday and Sunday). Tuesday is the day, when we see the most number of papers published.

![image](https://user-images.githubusercontent.com/111641830/236649946-2a3855e8-d0b6-41e8-987f-5dffc88c238f.png)

### Quantitative measures for the Authors present in the dataset :

| Authors                | Details  |
|------------------------|----------|
| Count                  | 252598   |
| Unique                 | 130748   |
| Top Author             | Yang  Li |
| Number of Publications | 82       |

### The below table shows us the Top 20 Authors with the maximum number of publications across all the departments :

| Author                 | Number of Publications |
|------------------------|-------|
| Yang    Li             | 82    |
| Yang    Liu            | 73    |
| Ilias    Diakonikolas  | 67    |
| Hao    Wang            | 65    |
| Xin    Wang            | 64    |
| Xi    Chen             | 64    |
| Oliver    Knill        | 61    |
| Krishnendu  Chatterjee | 60    |
| Iasson    Karafyllis   | 59    |
| Yu    Wang             | 57    |
| Scott    Aaronson      | 56    |
| Yu    Zhang            | 55    |
| Jaydip    Sen          | 54    |
| Andris    Ambainis     | 52    |
| Laszlo    Gyongyosi    | 51    |
| Wei    Wang            | 51    |
| Joseph Y.    Halpern   | 51    |
| Haris    Aziz          | 50    |
| Noga    Alon           | 47    |
| Marcus    Hutter       | 47    |

The below graph below shows us the distribution of number of authors for each research paper. As we can see, most of the papers have less than 10 authors, but some of them even had more than 100. In fact the highest number of authors was found to be 445 for one of the papers. The paper is linked [here](https://arxiv.org/abs/2206.04615)

![image](https://user-images.githubusercontent.com/111641830/236695995-09dcafcf-af27-470b-83f3-98d00bbef8c6.png)

The second one was to find out who the most contributing authors are. We see there are authors above 350 papers. Also we see that some of these are pioneers of AI like Yoshua Bengio

![image](https://user-images.githubusercontent.com/111641830/236696009-98421cda-3146-4fc3-839a-0225a6c1251c.png)

Third point we observe was related to our output variable: categories. We see that a paper can have up to 7 categories tagged, with most of the papers falling at 2 categories. Mean was found to be 1.9

![image](https://user-images.githubusercontent.com/111641830/236696016-464d6307-53f2-4780-8204-155cfdb03e44.png)

### The blelow graphs gives us the  idea of the most popular categories of work:

The next step is to check for most common categories. We see that LG (Machine Learning) towers over all other categories, indicating that a lot of work is being done in machine learning.

![image](https://user-images.githubusercontent.com/111641830/236650510-d88323ad-22ad-4522-9e80-171b18e5f245.png)

The next point will help us in determining what most common words are after the removal of stopwords. Words like 'data', 'model', 'learning' etc are found to be most common in *abstract*. These words will help our model learn that they are talking about machine learning or artificial intelligence. This also substantiates the above point of the EDA: most of the work in CS department is happening in machine learning

![image](https://user-images.githubusercontent.com/111641830/236696021-4a51eff8-a987-4e6e-9cb0-728bb36bcd58.png)

The next is most common words from the titles. Words like 'learning', 'networks' are again common, indicating involvement of machine learning. The word 'based' looks a bit odd here.

![image](https://user-images.githubusercontent.com/111641830/236696030-369d8116-34e1-434e-a0cd-e320526fe0ab.png)

Now that we have a clear understadning about the dataas we looked through the quantiative and descriptive analysis, we are ready to peform the sentiment analysis

## Sentiment Analysis: 

Initially, the data was used to create distinct sentiment classes

![image](https://user-images.githubusercontent.com/111655952/236695247-7dc22dfd-b40a-4311-920f-86a64536a5ae.png)  ![image](https://user-images.githubusercontent.com/111655952/236695255-559427fa-0ca5-40a1-838b-dc48443f4b55.png)

![image](https://user-images.githubusercontent.com/111655952/236695381-c7521bb6-ed61-433e-8510-d3b4dd063558.png)  ![image](https://user-images.githubusercontent.com/111655952/236695386-24f5e250-9eb6-4250-a5ea-ba767f1b48ab.png)

The positive sentiment is the majority class in all these analyses. A Vader pretrained model was used to mine all these sentiments in the abstracts. 

For example, take the last 2 records:

![image](https://user-images.githubusercontent.com/111655952/236695418-de835e12-cb6e-4de1-aee0-fe135278fb6f.png)

We indeed find that it is a positive sentiment talking about what the paper has developed better than the existing literature. Also, it’s a strong argument that’s also very technical and involved the use of some jargon. So, our model did a decent job of mining the sentiment classes. 

We also identify the technical sentiment through this word cloud visualization:

![image](https://user-images.githubusercontent.com/111655952/236695508-d93f59a6-13a1-4075-a43a-b9653b296d61.png)

All the top terms used for the technical class talk about the technical advancements and nuances of the paper. 

### Topic Analysis:

We extracted the top 5 topics from the classes overall:

![image](https://user-images.githubusercontent.com/111655952/236695528-6cdac21c-319d-4f33-8f24-5f6ee4d52b34.png)

Here we see that the top 5 topics are all containing technical words that fall under the positive class which is the majority class at around 80 %.  This again shows that most papers are highly technical and talk about the advancements and positive improvements brought about by the introduced paper.

#### Classification model – base:

![image](https://user-images.githubusercontent.com/111655952/236695552-e39519c3-3858-43ca-a4f1-1e54eb11111f.png)

We further went on to build a basic random forest classification model to predict the types of papers in the positive sentiment classes that are in the majority. We achieve barely acceptable accuracy. This shows that our model may be overfitting. So, we need to introduce more data for better generalization. So, for our overall modeling process, we use both positive and negative data.

### Named Entity Recognition: 

For our NER process, we found that the abstract text contained no location-based data. We were unable to extract geographic-tagged information. This again makes sense because there is very little geo-info when it comes to highly technical papers.

We were however able to extract the top nouns, verbs, and adjectives from the data as shown below. This again consolidates what we found earlier about the type of data in the majority classes. These nouns, verbs, and adjectives all convey a technical, positive (talks about the advancements of the paper), argumentative (existing methods minuses vs introduced methods positives) approach.

![image](https://user-images.githubusercontent.com/111655952/236695597-03773c7b-0eea-4843-bcee-43f5f533557d.png)  ![image](https://user-images.githubusercontent.com/111655952/236695602-f032bf09-822b-472e-93a4-36c4b34efccc.png)  ![image](https://user-images.githubusercontent.com/111655952/236695608-c02c95ef-8fc9-40a5-81c3-1751eccc87ba.png)



