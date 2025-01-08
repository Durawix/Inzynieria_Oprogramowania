import chromadb
import ollama
from read_data import read_data
import json 

data = read_data("data\store\Store_Inventory.xlsx")
database = "Store_Inventory"
client = chromadb.HttpClient(host='localhost', port=8000)

#Remove old collection from database
try:
     client.delete_collection(database)
except:
     pass
#Create new collection
collection = client.create_collection(database)

# Calculate embeddings for the 'Product' and 'Product Category' column
embeddings = []
for ind, row in data.iterrows():
  produkt = row["Product"]
  kategoria = row["Product Category"]
  aisle_name = row["Aisle Name"]
  data_to_emb = json.dumps({"Product": produkt, "Product Category": kategoria, "Aisle Name":aisle_name})
  print("Calculating embedding for", data_to_emb)
  response = ollama.embeddings(model="mxbai-embed-large", prompt=data_to_emb)
  embeddings.append(response["embedding"])
# add embeddings do dataframe
data["embeddings"] = embeddings
# Add embeddings and other columns to the collection
for i, row in data.iterrows():
  info_columns = ['Product', 'Price', 'Aisle Name', 'Aisle Number', 'Shelf', 'Product Category']
  embedding = row["embeddings"]
  document = json.dumps({col: row[col] for col in info_columns})
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[document]
  )
  
