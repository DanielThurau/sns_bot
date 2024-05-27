from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import tiktoken
import ast
from scipy import spatial
import sys


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit(1)


def strings_ranked_by_relatedness(
        query: str,
        dataframe: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float], list[str]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=embedding_model,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses_and_file = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]), row["file"])
        for i, row in dataframe.iterrows()
    ]
    strings_and_relatednesses_and_file.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses, files = zip(*strings_and_relatednesses_and_file)
    return strings[:top_n], relatednesses[:top_n], files[:top_n]


def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        dataframe: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses, files = strings_ranked_by_relatedness(query, dataframe)
    introduction = ('Use the below articles on the Service Nervous System to answer the subsequent question. If the '
                    'answer cannot be found in the articles, write "I could not find an answer."')
    question = f"\n\nQuestion: {query}"
    message = introduction
    for i, string in enumerate(strings):
        next_article = f'\n\nArticle:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            print("Using file in input {}".format(files[i]))
            message += next_article
    print()
    return message + question


def ask(
        query: str,
        dataframe: pd.DataFrame,
        model: str = "gpt-3.5-turbo",
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, dataframe, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the ServiceNervousSystem."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    embedding_model = "text-embedding-3-small"
    embedding_encoding = "cl100k_base"
    max_tokens = 8000  # the maximum for text-embedding-3-small is 8191
    GPT_MODEL = "gpt-3.5-turbo"

    embeddings_path = "embeddings.csv"
    df = pd.read_csv(embeddings_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

    if len(sys.argv) == 2:  # If there is one command-line argument
        print(ask(read_file(sys.argv[1]), df, GPT_MODEL))
    else:
        while True:
            input_line = input("Please enter a query: ")
            print()
            print(ask(input_line, df, GPT_MODEL))
            print()
