import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def get_embedding(text: str):
    """
    Generate an embedding using Gemini.
    Returns a Python list[float].
    """

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values