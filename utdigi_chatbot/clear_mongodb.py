# clear_mongodb.py — Utility to clear MongoDB collection
from pymongo import MongoClient

def clear_scraped_data():
    """Delete all records from scraped_data collection in utdallas_db_mini."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["utdallas_db_mini"]

    result = db["scraped_data"].delete_many({})
    print(f"✅ Cleared scraped_data — {result.deleted_count} documents deleted.")

if __name__ == "__main__":
    clear_scraped_data()
