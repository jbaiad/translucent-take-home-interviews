import argparse
from functools import cache
from openai import OpenAI
import pandas as pd
import pathlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = pathlib.Path(__file__).parent / "data" / "denials.csv"


def load_docs():
    df = pd.read_csv(DATA_PATH)

    docs = []
    for _, row in df.iterrows():
        doc = f"""
            Department: {row['department']}
            Denial Reason: {row['denial_reason']}
            Status: {row['status']}
            Date: {row['service_date']}
            Payer: {row['payer']}
        """
        docs.append(doc)

    return docs, df

@cache
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_docs(question: str, docs: list[str], df: pd.DataFrame):
    model = load_model()

    # Get embeddings for both the question and the documents
    question_embedding = model.encode([question])
    doc_embeddings = model.encode(docs)

    # Get the similar documents above a similarity threshold
    similarities = cosine_similarity(question_embedding, doc_embeddings).flatten()
    relevant_indices =  [i for i in range(len(similarities)) if similarities[i] > 0.50]
    print(f"FOUND {len(relevant_indices)} relevant documents.")

    return df.iloc[relevant_indices]

def answer(question: str) -> str:
    docs, df = load_docs()
    relevant_rows = get_relevant_docs(question, docs, df)

    oai_client = OpenAI()
    response = oai_client.responses.create(
        model="gpt-5-nano",
        input=f"""
        Your task is to answer the user's question based on the provided
        documents. Be factual and as succinct as possible.
        ---
        Question: {question}
        Relevant Claims Data: {relevant_rows.to_dict(orient='records')}
        Answer:
        """
    )

    return response.output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(answer(args.question))
