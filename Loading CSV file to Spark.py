## LOADING CSV TO SPARK ##

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import regexp_replace, col

# Create Spark session
spark = SparkSession.builder \
    .appName("Load CSV into Spark") \
    .getOrCreate()

# Path to your CSV file
csv_path = "/Users/anustha/Desktop/GROUP 6 BIG DATA/wwdc2025_comments.csv"  

# Read CSV with proper options for multiline and escaping quotes
df = spark.read.csv(
    csv_path,
    header=True,        
    multiLine=True,     
    escape='"',         
    inferSchema=True   )

# Clean up unwanted tabs or carriage returns, but keep newlines
df_clean = df.withColumn("body", regexp_replace(col("body"), "[\r\t]+", " ")) \
             .withColumn("readable_time", regexp_replace(col("readable_time"), "[\r\t]+", " "))

# Show data without truncation, to see full column contents clearly
df_clean.show(20, truncate=True)

# Print schema to verify data types
df_clean.printSchema()

# 1. Show null or empty fields
print("\n Null or empty fields:")
df_clean.filter(
    col("comment_id").isNull() |
    col("author").isNull() |
    col("body").isNull() |
    (col("body") == "") |
    col("score").isNull()
).show(truncate=False)

# 3. Count rows and authors
print(f"\n Total rows: {df_clean.count()}")
print(f" Unique authors: {df_clean.select('author').distinct().count()}")

# 4. Data type cleanup
df_clean = df_clean.withColumn("score", col("score").cast("int"))
df_clean = df_clean.withColumn("readable_time", to_timestamp("readable_time", "yyyy-MM-dd HH:mm:ss"))

# 5. Print schema
print("\n Final schema:")
df_clean.printSchema()

# Stop the Spark session when done
spark.stop()
