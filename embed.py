from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import tiktoken
import csv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

data_files = [
    "data/index.mdx",
    "data/integrating/frontend-integration.mdx",
    "data/integrating/index.mdx",
    "data/integrating/index-integration.mdx",
    "data/integrating/ledger-integration.mdx",
    "data/introduction/dao-alternatives.mdx",
    "data/introduction/sns-architecture.mdx",
    "data/introduction/sns-intro-high-level.mdx",
    "data/introduction/sns-launch.mdx",
    "data/launching/index.mdx",
    "data/launching/launch-steps-1proposal.mdx",
    "data/launching/launch-summary-1proposal.mdx",
    "data/managing/cycles-usage.mdx",
    "data/managing/making-proposals.mdx",
    "data/managing/manage-sns-intro.mdx",
    "data/managing/managing-nervous-system-parameters.mdx",
    "data/managing/sns-asset-canister.mdx",
    "data/testing/testing-before-launch.mdx",
    "data/testing/testing-locally.mdx",
    "data/testing/testing-on-mainnet.mdx",
    "data/tokenomics/index.mdx",
    "data/tokenomics/predeployment-considerations.mdx",
    "data/tokenomics/preparation.mdx",
    "data/tokenomics/rewards.mdx",
    "data/tokenomics/sns-checklist.mdx",
    "data/tokenomics/tokenomics-intro.mdx",
    "data/elna_sns_init.yaml",
]

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

data = []

for file_path in data_files:
    with open(file_path, "r") as file:
        data.append(file.read().replace("\n", ""))

encoding = tiktoken.get_encoding(embedding_encoding)
failed_length = False
for i, text in enumerate(data):
    text_encoding = encoding.encode(text)
    if len(text_encoding) > max_tokens:
        print("Text too large for file {}. Using {} / {} tokens".format(data_files[i], len(text_encoding), max_tokens))
        failed_length = True

if failed_length:
    exit("Max_tokens exceeded on prepared input")

out = [["file", "text", "embedding"]]
for i, text in enumerate(data):
    out.append([data_files[i], text, get_embedding(text, embedding_model)])

with open("embeddings.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(out)
# embedding = get_embedding(data)
# print(embedding)
