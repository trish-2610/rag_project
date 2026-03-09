from loader import load_files
from splitter import split_documents
docs = load_files("../dataset")
split_documents(docs)
