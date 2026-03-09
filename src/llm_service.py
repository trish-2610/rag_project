from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

## LLM 
try :
    llm = ChatGroq(
        model = "llama3-70b-8192",
        temperature = 0,
        api_key = os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Error in loading model : {str(e)} ")

def generate_response(query,context):
    """This function calls the LLM and answers the user query"""

    prompt = f"""
    Generate the answer of the user query using the given context only
    Context : {context}
    Query : {query}
    Answer : 
    """

    response = llm.invoke(prompt)
    return response.content