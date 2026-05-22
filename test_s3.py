from dotenv import load_dotenv
load_dotenv()

import boto3

s3 = boto3.client("s3")

print("Buckets:")

for bucket in s3.list_buckets()["Buckets"]:
    print(bucket["Name"])