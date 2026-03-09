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
    You are an expert assistant specialized in analyzing government policies, AI strategies and public sector reports.

    Your task is to answer the user's question using ONLY the information provided in the context below.

    Important Instructions to follow :
    1. Do NOT use any external knowledge.
    2. Base your answer strictly on the provided context.
    3. If the context does not contain the answer, say:
    "The answer is not available in the provided documents."
    4. Provide clear and factual responses.
    5. When possible, mention the policy name, scheme name or organization referenced in the context.

    Context : {context}

    User Question : {query}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content