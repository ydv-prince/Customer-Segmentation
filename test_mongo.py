from dotenv import load_dotenv
load_dotenv()

from src.configuration.mongodb_connection import MongoDBClient

db = MongoDBClient(database_name="customer_segmentation_db")

print("Connected")

collection = db.database["customer_profiles"]

print(collection.count_documents({}))