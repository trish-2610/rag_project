from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

def load_files(dataset_path):
    """This function will load all PDF files from dataset directory and attach the metadata"""

    documents = []
    pdf_files = list(Path(dataset_path).glob("**/*.pdf"))
    print(f"{len(pdf_files)} files loaded")
    

    for pdf in pdf_files:
        loader = PyMuPDFLoader(str(pdf))
        docs = loader.load()

        ## category from folder name
        category = pdf.parent.name

        for d in docs:
            d.metadata["source"] = pdf.name
            d.metadata["category"] = category ## pdf.parent.name
             
        documents.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf.name}")
        
    return documents
    
