from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, regexp_replace, col, udf, window
from pyspark.sql.types import StringType
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create Spark session
spark = SparkSession.builder.appName("NLPBatch").getOrCreate()

# Load CSV 
csv_path = "/Users/anustha/Desktop/GROUP 6 BIG DATA/wwdc2025_comments.csv"
df = spark.read.csv(csv_path, header=True, multiLine=True, escape='"', inferSchema=True)

# Cleaning the column
df_clean = df.withColumn("body", regexp_replace(col("body"), "[\r\t]+", " ")) \
             .withColumn("readable_time", regexp_replace(col("readable_time"), "[\r\t]+", " ")) \
                .withColumn("score", col("score").cast("int")) \
                    .withColumn("readable_time", to_timestamp("readable_time", "yyyy-MM-dd HH:mm:ss"))

#UDF for sentiment
def clean_text(text):
    if text:
        return text.lower()
    else:
        return ""

def get_sentiment(text):
    if text:
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return "positive"
        elif analysis.sentiment.polarity < 0:
            return "negative"
        else:
            return "neutral"
    else:
        return "neutral"
    
clean_text_udf = udf(clean_text, StringType())
sentiment_udf = udf(get_sentiment, StringType())

# Apply UDFs
df_nlp = df_clean.withColumn("cleaned_body", clean_text_udf(col("body"))) \
                 .withColumn("sentiment", sentiment_udf(col("cleaned_body")))



### TEXT PREPROCESSING AND MODEL TRAINING

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# tokenize
tokenizer = Tokenizer(inputCol="cleaned_body", outputCol="words")
df_tokenized = tokenizer.transform(df_nlp)

# remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# tf -idf (Term Frequency - Inverse Document Frequency)
#It's a numerical statistic used in Natural Language Processing (NLP) and information retrieval to reflect how important a word is to a document in a collection or corpus.

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
tf_data = hashingTF.transform(df_filtered)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(tf_data)
tfidf_data = idf_model.transform(tf_data)

# Output with features
tfidf_data.select("readable_time", "cleaned_body", "sentiment", "features").show(truncate=60)





from pyspark.sql.functions import window

# Apply window to each comment
comment_with_window = tfidf_data.withColumn(
    "window", window(col("readable_time"), "60 minutes")
)

# Extract window_start and window_end from window column
tfidf_output_df = comment_with_window.select(
    col("window").start.alias("window_start"),
    col("window").end.alias("window_end"),
    "cleaned_body",
    "sentiment"
).orderBy("window_start")


tfidf_output_df.show(100, truncate=60)






# Set output path
output_path_tfidf = "/Users/anustha/Desktop/GROUP 6 BIG DATA/kkk"

# Save DataFrame to a single, clean CSV
tfidf_output_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("delimiter", ",") \
    .csv(output_path_tfidf)

print(f"TF-IDF data (without features) saved to: {output_path_tfidf}")








# Show some results
df_nlp.select("body", "cleaned_body", "sentiment").show(20, truncate=100)

# Show sentiment counts 
df_nlp.groupBy("sentiment").count().show()

# Aggregate sentiment counts over 60-minute windows
agg_df = df_nlp.groupBy(
    window(col("readable_time"), "60 minutes"),
    col("sentiment")
).count()

# Pivot sentiment values (positive, neutral, negative) to columns
pivot_df = agg_df.groupBy("window").pivot("sentiment").sum("count")

# Add window start and end as separate columns for clarity
result_df = pivot_df.select(
    col("window").start.alias("window_start"),
    col("window").end.alias("window_end"),
    col("negative"),
    col("neutral"),
    col("positive")
).orderBy("window_start")

# Show results
result_df.show(100, truncate=False)


# --- Save aggregated sentiment counts to CSV ---
output_path = "/Users/anustha/Desktop/GROUP 6 BIG DATA/sentiment_over_time_pivoted"

result_df.coalesce(1) \
         .write \
         .mode("overwrite") \
         .option("header", "true") \
         .csv(output_path)

print(f"Pivoted sentiment data saved to: {output_path}")

# --- Collect for visualization ---
# Convert Spark DataFrame to Pandas DataFrame for plotting
pdf = agg_df.select(
    col("window").start.alias("window_start"),
    col("sentiment"),
    col("count")
).toPandas()

# Convert result_df (pivoted sentiment counts) to Pandas for plotting
pdf_sentiment = result_df.toPandas()

# Format datetime for x-axis labels
pdf_sentiment['window_start_str'] = pdf_sentiment['window_start'].dt.strftime('%Y-%m-%d %H:%M')

# Set up bar width and positions
bar_width = 0.25
x = np.arange(len(pdf_sentiment))

plt.figure(figsize=(14, 7))

# Plot bars for each sentiment
plt.bar(x - bar_width, pdf_sentiment['negative'].fillna(0), width=bar_width, color='red', label='Negative')
plt.bar(x, pdf_sentiment['neutral'].fillna(0), width=bar_width, color='gray', label='Neutral')
plt.bar(x + bar_width, pdf_sentiment['positive'].fillna(0), width=bar_width, color='green', label='Positive')

# Label the x-axis with window start times
plt.xticks(x, pdf_sentiment['window_start_str'], rotation=45, ha='right')

# Titles and labels
plt.xlabel('Time Window Start')
plt.ylabel('Number of Comments')
plt.title('Sentiment Counts Over Time (60-minute windows)')
plt.legend()

plt.tight_layout()
plt.grid(axis='y')
plt.show()

## Voilume Over Time ##
# --- Comment volume aggregation over 10-minute windows ---
volume_df = df_clean.groupBy(
    window(col("readable_time"), "10 minutes")
).count().orderBy("window")

# Show comment volume over time
volume_df.select(
    col("window").start.alias("window_start"),
    col("window").end.alias("window_end"),
    "count"
).show(100, truncate=False)

# Save comment volume to CSV
volume_output_path = "/Users/anustha/Desktop/GROUP 6 BIG DATA/comment_volume_over_time"
volume_df.select(
    col("window").start.alias("window_start"),
    col("window").end.alias("window_end"),
    "count"
).coalesce(1).write.mode("overwrite").option("header", "true").csv(volume_output_path)

print(f"Comment volume data saved to: {volume_output_path}")

# --- Visualization of comment volume over time ---
volume_pdf = volume_df.select(
    col("window").start.alias("window_start"),
    "count"
).toPandas()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(volume_pdf['window_start'], volume_pdf['count'], marker='o')
plt.title("Comment Volume Over Time (10-minute windows)")
plt.xlabel("Time Window Start")
plt.ylabel("Number of Comments")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Display sample comments with sentiment in formatted output ---
sample_comments = df_nlp.select("body", "sentiment").limit(5).collect()

print("\nComment\tSentiment")
for row in sample_comments:
    comment = row['body'].strip().replace('\n', ' ')
    sentiment = row['sentiment'].capitalize()
    print(f'"{comment}"\t{sentiment}')

# --- Save formatted comments and sentiments to CSV ---
# Select desired columns and limit rows (remove .limit(5) if you want the full set)
formatted_df = df_nlp.select("body", "sentiment")

# Convert to Pandas for easier CSV export
formatted_pdf = formatted_df.toPandas()

# Capitalize sentiment and clean newlines
formatted_pdf['sentiment'] = formatted_pdf['sentiment'].str.capitalize()
formatted_pdf['body'] = formatted_pdf['body'].str.replace('\n', ' ').str.strip()

# Save to CSV
formatted_output_path = "/Users/anustha/Desktop/GROUP 6 BIG DATA/comments_with_sentiment.csv"
formatted_pdf.to_csv(formatted_output_path, index=False)

print(f' Formatted comments saved to: {formatted_output_path}')
 
# Stop Spark session
spark.stop()
