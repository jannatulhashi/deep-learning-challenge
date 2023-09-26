# Optimizing Alphabet Soup's Charitable Investments with Deep Learning

The project uses deep learning to help Alphabet Soup predict which funded organizations are most likely to succeed, based on historical data of over 34,000 previous fund recipients. This allows the foundation to allocate funds more effectively.

# Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- **EIN and NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special considerations for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

### Instructions
#### Step 1: Preprocess the Data

Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I’ll need to preprocess the dataset. This step prepares me for Step 2, where I'll compile, train, and evaluate the neural network model. I'll complete the following preprocessing steps:
1: I'll read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in my dataset:

   - What variable(s) are the target(s) for my model?
   - What variable(s) are the feature(s) for my model?

2: Drop the EIN and NAME columns.

3: Determine the number of unique values for each column.

4: For columns that have more than 10 unique values, determine the number of data points for each unique value.

5: I'll use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then I'll check if the binning was successful.

6: Then I'll use pd.get_dummies() to encode categorical variables.

7: Split the preprocessed data into a features array, X, and a target array, y. I'll use these arrays and the train_test_split function to split the data into training and testing datasets.

8: Then, I'll scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

#### Step 2: Compile, Train, and Evaluate the Model

Using my knowledge of TensorFlow, I’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I’ll need to think about how many inputs there are before determining the number of neurons and layers in my model. Once I’ve completed that step, I’ll compile, train, and evaluate my binary classification model to calculate the model’s loss and accuracy.

1: Continue using the file in Google Colab in which I performed the preprocessing steps from Step 1.

2: Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3: Create the first hidden layer and choose an appropriate activation function.

4: If necessary, add a second hidden layer with an appropriate activation function.

5: Create an output layer with an appropriate activation function.

6: Check the structure of the model.

7: Compile and train the model.

8: Create a callback that saves the model's weights every five epochs.

9: Evaluate the model using the test data to determine the loss and accuracy.

10: Save and export my results to an HDF5 file. I'll name the file **AlphabetSoupCharity.h5**.

#### Step 3: Optimize the Model

Using my knowledge of TensorFlow, optimized my model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize my model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:

     - Dropping more or fewer columns.
     - Creating more bins for rare occurrences in columns.
     - Increasing or decreasing the number of values for each bin.
     - Add more neurons to a hidden layer.
     - Add more hidden layers.
     - Use different activation functions for the hidden layers.
     - Add or reduce the number of epochs to the training regimen.

1: I'll create a new Google Colab file and name it **AlphabetSoupCharity_Optimization.ipynb**.

2: Import my dependencies and read in the **charity_data.csv** to a Pandas DataFrame.

3: I'll preprocess the dataset as I did in Step 1. And I'll make sure to adjust for any modifications that came out of optimizing the model.

4: I'll design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5: Then, I'll save and export my results to an HDF5 file. and I'll name the file **AlphabetSoupCharity_Optimization.h5**.

#### Step 4: Write a Report on the Neural Network Model

I'll write a report on the performance of the deep learning model I created for Alphabet Soup.
The report should contain the following: 

1: **Overview of the analysis:** Explain the purpose of this analysis.

2: **Results:** Using bulleted lists and images to support my answers, address the following questions:

- Data Preprocessing
    - What variable(s) are the target(s) for my model?
    - What variable(s) are the features for my model?
    - What variable(s) should be removed from the input data because they are neither targets nor features?

- Compiling, Training, and Evaluating the Model
    - How many neurons, layers, and activation functions did I select for my neural network model, and why?
    - Were I able to achieve the target model performance?
    - What steps did I take in my attempts to increase model performance?

3: **Summary:** I'll summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then I'll explain my recommendation.

## Report: Alphabet Soup Funding Analysis Report

**Overview of the Analysis:**

The primary objective of this analysis is to leverage the power of machine learning, particularly deep learning, to create a binary classification model that can predict the success of applicants funded by Alphabet Soup. This tool will assist Alphabet Soup in making informed decisions about which organizations to fund based on the likelihood of their success.

**Results:**

**Data Preprocessing:**

**Target Variable:**

**IS_SUCCESSFUL:** This indicates whether the funds provided by Alphabet Soup were used effectively by the organization.

**Features:**

All the columns present in the dataset, except for EIN and IS_SUCCESSFUL, were used as features. This includes APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, and NAME.

**Variables Removed:**

**EIN:** This column was removed as it is an identification column and does not provide any significant information to determine the success of an organization.

**Compiling, Training, and Evaluating the Model:**

**Model Structure:**

Input Layer: 128 neurons with ReLU activation function.

First Hidden Layer: 64 neurons with ReLU activation function.

Second Hidden Layer: 32 neurons with ReLU activation function.

Third Hidden Layer: 16 neurons with ReLU activation function.

Output Layer: 1 neuron with Sigmoid activation function since it's a binary classification problem.

**Model Compilation:**

Optimizer: Adam

Loss function: Binary Crossentropy

Metrics: Accuracy

**Model Performance:**

The model's accuracy on the test set was found to be 79%.

**Steps to Optimize Model Performance:**

The data was preprocessed by binning values of categorical columns that had a large number of unique values. Specifically, values in NAME, APPLICATION_TYPE, and CLASSIFICATION columns were grouped into an 'Other' category if they occurred fewer times than a set cutoff value.
Data was then converted into a numerical format using one-hot encoding (pd.get_dummies).

The data was standardized using StandardScaler to have zero mean and unit variance.

The deep learning model was designed with multiple layers and neurons to capture the underlying patterns in the data. The model was trained over 100 epochs.

**Summary:**

The deep learning model designed for predicting the success of applicants funded by Alphabet Soup provides a mechanism for understanding the key factors that determine whether an organization will use the funds effectively. The model's accuracy indicates its effectiveness in making these predictions.

However, deep learning models can sometimes be seen as a 'black box', and it might be challenging to interpret which features had the most influence on the predictions.

**Recommendation:**

A different approach could be to use ensemble learning methods, such as Random Forest. This model can provide insights into feature importance, allowing Alphabet Soup to understand which factors are the most critical in determining an organization's success. This is also less prone to overfitting compared to deep neural networks and can sometimes achieve comparable or even better performance with the right hyperparameter tuning.

#### Step 5: Copy Files Into Your Repository

Now that I've finished with my analysis in Google Colab, I need to get my files into my repository for final submission.

- I'll download my Colab notebooks to my computer.

- I'll move them into my Deep Learning Challenge directory in my local repository.

- And, then I'll push the added files to GitHub.

***References***

IRS. Tax Exempt Organization Search Bulk Data Downloads.
[](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads)
