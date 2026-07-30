# Data Pipeline Design

## Raw Data Handler

### Design Decisions

1. **Data Extraction**: We use pandas to read CSV and Parquet files, and the built-in json module for JSON files. This allows for efficient handling of different file formats.

2. **Data Transformation**:
   - Merge dataframes based on common keys (cc_num and trans_num).
   - Standardize column names for consistency.
   - Convert date columns to datetime for easier time-based feature extraction.
   - Create derived features (hour, day_of_week, month) for time-based analysis.
   - Handle missing values in the 'is_fraud' column by filling with 0 (assuming non-fraudulent by default).

3. **Data Description**: We provide a comprehensive description of the dataset, including shape, columns, data types, missing values, and fraud ratio.

4. **Data Loading**: We save the processed data in Parquet format for efficient storage and quick reading in subsequent steps.

### Rationale

- The merging strategy allows us to combine all relevant information from different sources.
- Time-based features are crucial for fraud detection, as fraudulent activities often follow temporal patterns.
- Standardizing column names and handling missing values ensures data consistency and prevents issues in downstream processes.
- Using Parquet format for storage provides a good balance between compression and read performance.

## Dataset Designer

### Design Decisions

1. **Data Extraction**: We read the processed raw data from the Parquet file created by the Raw Data Handler.

2. **Data Sampling**:
   - We use a 80-20 split for train-test.
   - We use stratified sampling to maintain the same fraud ratio across all splits.

3. **Data Description**: We provide information about the shape and fraud ratio for each partition.

4. **Data Loading**: We save each partition (train, validation, test) as separate Parquet files.

### Rationale

- The 80-20 split is a common practice that provides a good balance between having enough training data and a representative test set.
- Stratified sampling ensures that each partition has a similar distribution of fraudulent and non-fraudulent transactions, which is crucial for model training and evaluation.
- Saving partitions separately allows for easy access in subsequent steps of the machine learning pipeline.

## Feature Extractor

### Design Decisions

1. **Data Extraction**: We read the partitioned data (train, validation, test) from the Parquet files created by the Dataset Designer.

2. **Feature Transformation**:
   - Time-based features: We use sine and cosine transformations for hour and day of week to capture cyclical patterns.
   - Transaction amount: We apply log transformation to handle the wide range of transaction amounts.
   - Merchant category: We encode categories as integers.
   - Location-based features: We calculate the Haversine distance between the customer's location and the merchant's location.
   - Feature scaling: We apply standard scaling to normalize feature ranges.

3. **Data Description**: We provide information about the shape, columns, and data types for each feature set and target variable.

### Rationale

- Sine and cosine transformations for time features allow the model to capture cyclical patterns without imposing an arbitrary linear relationship.
- Log transformation of transaction amounts helps to handle the wide range and potential skewness in the data.
- Encoding merchant categories allows the model to work with this categorical data.
- Haversine distance provides a meaningful measure of the distance between the customer and merchant, which could be indicative of fraudulent activity.
- Standard scaling ensures that all features are on a similar scale, which is important for many machine learning algorithms.
