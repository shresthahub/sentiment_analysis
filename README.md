# Apple WWDC 2025 Reddit Sentiment Analysis
This project analyzes Reddit user sentiment around Apple's WWDC 2025 event using Reddit comments. It extracts data, processes it with PySpark, performs sentiment analysis, and visualizes trends over time.

# Setup Instructions
Make sure you have the following installed:

- Python 3.x
- Apache Spark (`pyspark`)
- `textblob`
- `matplotlib`
- `pandas`
- `numpy`

### 1. Data Sourcing
- **Source**: Reddit (r/apple and other related subreddits)
- **Tool**: [PRAW](https://praw.readthedocs.io/) (Python Reddit API Wrapper)
- **Access Method**: Reddit API
- **Data Collected**:
  - `comment_id`
  - `author`
  - `body` (comment text)
  - `readable_time`
  - `score`

### 2. Data Processing
- **Tool**: Apache Spark (PySpark)
- **Steps**:
  - Reading multi-line CSV files with escape character handling
  - Removing unwanted special characters (`\t`, `\r`)
  - Handling `null` and empty values
  - Data type conversions (timestamp casting, score to int)
  - Storing cleaned data locally for reuse
 
### 3. Sentiment Analysis
- **Tool**: TextBlob via UDFs in PySpark
- **Process**:
  - Custom UDFs for:
    - Text normalization (lowercasing, cleaning)
    - Sentiment classification
  - **Classification Labels**:
    - `Positive`
    - `Neutral`
    - `Negative`

### 4. Data Visualization
- **Tools**:
  - `matplotlib` (Python) for time-series plotting
  - `Excel` for polished charts and tabular presentation
- **Visual Outputs**:
  - **Sentiment Counts Over Time** (in 60-minute windows)
  - **Comment Vol Over Time** (in 10-minute windows)
 
### Additional NLP Techniques
- Tokenization and Stopword Removal
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Feature Engineering using Spark ML pipeline
- Time windowing using Spark SQL functions for better temporal granularity

# Future Works
- Integrate real-time data streaming using Kafka and Spark Structured Streaming.
- Improve sentiment analysis with MLlib or transformer-based NLP models.
- Deploy the pipeline on HDFS with Hive for scalable storage and querying.

# Group Contributions
- Anusha Shrestha: worked on code and research paper
- Bruna Almeida: worked on research paper and presentatiion
- Maria De Paiva: worked on visualizations, research paper and presentation
- Pratik Maharjan: worked on presentation and research paper
- Zhiqi Zhang: worked on research paper and presentation
