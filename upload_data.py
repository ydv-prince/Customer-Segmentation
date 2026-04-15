import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()

# Retrieve URI from environment variable
uri = os.getenv("MONGO_DB_URL")
if not uri:
    raise ValueError("Environment variable MONGO_DB_URL is not set!")

print("=" * 60)
print("Connecting to MongoDB...")

client = MongoClient(uri)

try:
    client.admin.command("ping")
    print("✅ Connected successfully.")
except Exception as e:
    print("❌ Connection failed:", e)
    exit()

print("=" * 60)
print("Databases before insertion:")
print(client.list_database_names())

db = client["customer_segmentation_db"]
print("\nUsing Database:", db.name)

collection = db["customer_profiles"]
print("Using Collection:", collection.name)

print("=" * 60)
print("Reading CSV...")

df = pd.read_csv("notebooks/marketing_campaign.csv", sep="\t")

print("CSV Shape:", df.shape)
print(df.head())

print("=" * 60)
print("Documents BEFORE delete:", collection.count_documents({}))

collection.delete_many({})

print("Documents AFTER delete:", collection.count_documents({}))

print("=" * 60)
print("Inserting documents...")

result = collection.insert_many(df.to_dict("records"))

print("Inserted IDs:", len(result.inserted_ids))

print("=" * 60)
print("Documents AFTER insert:", collection.count_documents({}))

print("=" * 60)
print("Sample document from MongoDB:")

print(collection.find_one())

print("=" * 60)
print("Databases after insertion:")

print(client.list_database_names())

print("=" * 60)