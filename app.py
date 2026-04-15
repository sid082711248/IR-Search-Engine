from flask import Flask, render_template, request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = fetch_20newsgroups(subset='all')
documents = data.data[:500]
file_names = [f"Document_{i}" for i in range(len(documents))]

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(documents)

def search_engine(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)

    scores = list(enumerate(similarity[0]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in sorted_scores[:5]:
        results.append({
            "doc": file_names[i],
            "score": f"{score:.4f}",
            "preview": documents[i][:150]
        })
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form["query"]
        results = search_engine(query)

    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
