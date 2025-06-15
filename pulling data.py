### Apple WWDC 2025 Comments ###

import praw
import csv
from datetime import datetime

# Initialize Reddit instance with your credentials
reddit = praw.Reddit(
    client_id='0t5mf-wxscGqjc35hXqvJg',
    client_secret='0BNq0f_mW6s9NI6m7ndAUZpnJgH78g',
    user_agent='wwdc2025-sentiment-analysis-script by /u/Flashy-Chair9146'
)

# Reddit post URL
post_url = 'https://www.reddit.com/r/apple/comments/1l7ewbh/wwdc_2025_postevent_megathread/'

# Get the submission object from the URL
submission = reddit.submission(url=post_url)

# Load all comments (replace "MoreComments")
submission.comments.replace_more(limit=None)

# Print all comments
for comment in submission.comments.list():
    print(comment.body)
    print('-' * 80)

# Open a CSV file to write comments
with open('wwdc2025_comments.csv', mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow(['comment_id', 'author', 'body', 'score', 'created_utc', 'readable_time'])

    # Write comment data rows
    for comment in submission.comments.list():
        author = comment.author.name if comment.author else '[deleted]'
        created_utc = comment.created_utc
        readable_time = datetime.utcfromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S')
        print("Readable time:", readable_time)
        writer.writerow([comment.id, author, comment.body, comment.score, comment.created_utc, readable_time])

print("Comments saved to wwdc2025_comments.csv")