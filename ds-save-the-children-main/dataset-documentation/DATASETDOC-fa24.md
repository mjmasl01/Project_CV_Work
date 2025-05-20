***Project Information*** 

* What is the project name?  
    * Save The Children: Catch-Up Clubs 
* What is the link to your project’s GitHub repository?   
    * https://github.com/BU-Spark/ds-save-the-children  
* What is the link to your project’s Google Drive folder? \*\**This should be a Spark\! Owned Google Drive folder \- please contact your PM if you do not have access\*\**  
    * https://drive.google.com/drive/folders/1ttEss7nD3EMNkrQ3_vfn-Dt4O4SgqBGM?usp=sharing 
* In your own words, what is this project about? What is the goal of this project?   
    *  This project is about the outcome of Catch-Up Clubs: a pioneered approach to getting children safely back to school. It addresses key questions to identify predictors of student success within the CuC initiative. Specifically, it seeks to answer the best predictors of student progression in their literacy studies,  numeracy studies, Social Emotional Learning (SEL) studies, and retention (i.e., the prevention of student dropouts). This project should integrate predictive analysis into dashboards and advance their current analytics.  
* Who is the client for the project?  
    * Save the Children, the client for this project, is a leading international nonprofit organization dedicated to child welfare and development.
* Who are the client contacts for the project?  
    * mohini.venkatesh@savethechildren.org 
    * fitzsimmons.sean@savethechildren.org 
* What class was this project part of?
    * DS701 Tools for Data Science

***Dataset Information***

* What data sets did you use in your project? Please provide a link to the data sets, this could be a link to a folder in your GitHub Repo, Spark\! owned Google Drive Folder for this project, or a path on the SCC, etc.  
    * The datasets we used in our project are basically from surveys of students' behaviors and which are imported from csv files in microsoft sharepoint. The links are below:
        * [Primary Data ](https://savethechildren1.sharepoint.com/:f:/r/sites/BUSpark-Waliku/Shared%20Documents/General%20Information/Primary%20Data?csf=1&web=1&e=j2rorV)
        * [Supplementary Data](https://savethechildren1.sharepoint.com/:f:/r/sites/BUSpark-Waliku/Shared%20Documents/General%20Information/Supplementary%20Data?csf=1&web=1&e=KeqZEs)
* Please provide a link to any data dictionaries for the datasets in this project. If one does not exist, please create a data dictionary for the datasets used in this project. **(Example of data dictionary)**   
    * [CuC Data Dictionary](https://github.com/BU-Spark/ds-save-the-children/tree/dev?tab=readme-ov-file#data-dictionary)
* What keywords or tags would you attach to the data set?  
    * Domain(s) of Application: Data Analysis, Looker Studio Dashboard
    * Education
 

*The following questions pertain to the datasets you used in your project.*   
*Motivation* 

* For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description. 
    * The dataset was created to measure and discuss the outcome of their cuc program and find more ways to improve their current teaching. The specific task is to find best predictors for each crucial outcome like literacy ability, numeracy ability, sel score, retention. The gap that needed to be filled is that more data of other aspects of children themselves like family conditions, teaching differences could be added to the dataset. 

*Composition*

* What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? What is the format of the instances (e.g., image data, text data, tabular data, audio data, video data, time series, graph data, geospatial data, multimodal (please specify), etc.)? Please provide a description.   
    * A single instance in our dataset stands for a single student’s description and behavior through that program cycle. Data from 3 countries were used​. The format of the instances is basically tabular data.
    
* How many instances are there in total (of each type, if appropriate)?  
    * Nigeria: 15314 students
    * Uganda: 40202 students
    * Philippine: 2811 students
    * Total: 54814 students

* Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).  
    * Instances may come from a larger set, and the ‘childrenID’ field in the data table show that the data that the client provides to us is a subset of their source data.
    
* What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.   
    * Each instance represents a student. In the table, we can find the student's information on personal condition, school, class, and registration, the study status change data during the process of the CuC project.

* Is there any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include redacted text.   
    * Yes, there is missing data which is because  it was unavailable and not given by our clients.
    * The attendance data of approximately 60,000 students over two cycles, with less than 20,000 absent, has a cyclical pattern in the data, peaking in March and decreasing thereafter.
    * SEL data and numeracy data only have a single country.
    * Specific details about classes and schools are also missing.
 
* Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them  
    * Yes, Training Set (70-80%), Test Set(30-20%)
* Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.   
    * The dataset contains potential issues such as outliers in continuous variables like age, as well as a significant amount of missing data in additional data related to students' disability status. 
* Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources,   
  * Are there guarantees that they will exist, and remain constant, over time;  
      * The dataset is self-contained because it is collected using surveys. Since the data is directly gathered and stored within the dataset, it does not rely on external resources that may change or become unavailable over time.
      
  * Are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created)?  
      * No
  * Are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a dataset consumer? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points as appropriate.   
      * There are no external resources associated with the dataset, so no restrictions such as licenses or fees apply. 
* Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description.   
    * Yes, students' personal information should be private, and other features, such as residence conditions, traffic situations, etc., which could reveal the students’ family information and economic status, are not suitable for disclosure.
* Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.   
    * No
* Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.   
    * Nothing like that has happened.
* Dataset Snapshot, if there are multiple datasets please include multiple tables for each dataset. 


| Data | Size of dataset |
| :---- | :---- |
| Uganda | 104.9MB |
| Nigeria  | 24.9MB |
| Philippines | 5.17MB |


  
*Collection Process*

* What mechanisms or procedures were used to collect the data (e.g., API, artificially generated, crowdsourced \- paid, crowdsourced \- volunteer, scraped or crawled, survey, forms, or polls, taken from other existing datasets, provided by the client, etc)? How were these mechanisms or procedures validated?  
    * Our data sets are provided directly by our client, and we evaluate the data by walking through how each of the columns functions in the dataset with the client. As for the client dataset, the original data from the client were collected in the form of questionnaires. The client survey template can be accessed through this [link](https://kobo-ee.savethechildren.net/x/uUnFWuln). 
* If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?  
    * We took all the data from the client and applied all the data to our project analysis and modeling.
    
* Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. 
    * Literacy study in Uganda: 2022-2023
    * Numeracy study in Uganda: 2023-2024
    * Retention study in Uganda: 2021-2024
    * Literacy study in Nigeria: 2023
    * SEL study in Nigeria: 2024
    * Literacy study in Philippine: 2023
    * Retention study in Philippine: 2023 


*Preprocessing/cleaning/labeling* 

* Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remaining questions in this section.   
    * Missing Values: Removed empty records to ensure accurate analysis.
    * Repetitive Values: Remove Children that appear more than once 
    * Data Consistency: Verified that data types were appropriate for each variable.

* Were any transformations applied to the data (e.g., cleaning mismatched values, cleaning missing values, converting data types, data aggregation, dimensionality reduction, joining input sources, redaction or anonymization, etc.)? If so, please provide a description.   
    * Remove records that “Baseline to Endline” is blank.
    * Feature Encoding: Encode categorical variables into numerical variables like mapping ResultBaseline, ResultRound1, ResultRound2, ResultRound3, ResultEndline to numbers
    * Normalization: Normalize and scale features.
    * Cleaning: Remove outliers.

* Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data, this could be a link to a folder in your GitHub Repo, Spark\! owned Google Drive Folder for this project, or a path on the SCC, etc.  
    * No, there are privacy restrictions.
* Is the code that was used to preprocess/clean the data available? If so, please provide a link to it (e.g., EDA notebook/EDA script in the GitHub repository).
    * Yes, the preprocessing code can be found in our github. The files can be found in separate folders for different dataset: https://github.com/BU-Spark/ds-save-the-children  

*Uses* 

* What tasks has the dataset been used for so far? Please provide a description.
    * Key Questions:
What are the best predictors of student progression in their literacy studies?
What are the best predictors of student progression in their numeracy studies?
What are the best predictors of student progression in their Social Emotional Learning (SEL) studies?
What are the best predictors of student retention (i.e., the prevention of student dropouts)?
    * Sub-questions:
Uganda, Nigeria: What are the predictors for learners who stay at beginner or letter level even after one or two cycles of intervention?  
 Uganda, Nigeria: What is the likelihood for success for learners with disabilities (lot of difficulty or cannot do at all on one or more of the 6 disability domains)?   
For attendance as a predictor, what is the minimum level for achieving expected outcome of reading with comprehension? 
Uganda, Nigeria: Is the starting point of a learner's reading ability a predictor for the level of improvement expected? So are those at the lowest levels making more progress in the CuC cycle than those at higher levels? 
Nigeria: Are learners who receive SEL interventions (through VYA sessions) more likely to do better in their CuC club literacy outcomes than those who do not receive it? 

   
* What (other) tasks could the dataset be used for?  
    * Relationship between students progression and disability situation
    * Is in- or out-of-school status of a child a predictor for learner progression and graduation?

* Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?   
    * The dataset was sourced from surveys, which may introduce selection bias due to underrepresentation of certain groups.
Removing missing, duplicate, and outlier data may have caused systematic bias by excluding important edge cases.
Sensitive information in the dataset could pose privacy risks if anonymization measures were inadequate.

* Are there tasks for which the dataset should not be used? If so, please provide a description.
    * Tasks Involving Non-Child-Centered Applications
    * Sensitive or Confidential Data Sharing:


*Distribution*

* Based on discussions with the client, what access type should this dataset be given (eg., Internal (Restricted), External Open Access, Other)?
    * The dataset should be internal because it contains a lot of private information about our target children.


*Maintenance* 

* If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. 
    * Available via the survey template link provided above, the template contains all the fields relevant to this project.

*Other*

* Is there any other additional information that you would like to provide that has not already been covered in other sections?
    * Yes, we hope more numeracy and sel score data from different countries and different academic years can be provided. More school data as well as class data like location, environment, class size and teachers is also wanted.

