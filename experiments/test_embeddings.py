from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

sentences = [
    "query: Combien de jours  ai-je pour annuler un contrat signé à domicile",
    "query: How many days do I have to cancel a door-to-door contract?",
    "query: Quelle est la meilleure recette de tarte aux pommes ?",
]

embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)

# Dot product of query 1 (embeddings[0]) with query 2 (embedding[1]) and 3 (embeddings[2])
# cosine similarities (sim)
sim_01 = embeddings[0] @ embeddings[1]
sim_02 = embeddings[0] @ embeddings[2]

print(f"Similarities between query 0-1: {sim_01:.4f}")
print(f"Similarities between query 0-2: {sim_02:.4f}")
