from flask import Flask, render_template, request
import joblib
import os

from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# ---------------------------------------------------------------------
# Environment & API clients
# ---------------------------------------------------------------------
load_dotenv()  # loads .env from project root

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
if not gemini_api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")

# Groq (LLAMA) client
client = Groq(api_key=groq_api_key)

# Gemini client
genai.configure(api_key=gemini_api_key)

# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/main", methods=["GET", "POST"])
def main():
    return render_template("main.html")


@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    return render_template("dbs.html")


@app.route("/dbs_prediction", methods=["GET", "POST"])
def dbs_prediction():
    q = float(request.form.get("q"))
    model = joblib.load("DBS_SGD_model.pkl")
    r = model.predict([[q]])
    return render_template("dbs_prediction.html", r=r)


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    return render_template("chatbot.html")


# ---------------------------------------------------------------------
# LLAMA (Groq)
# ---------------------------------------------------------------------
@app.route("/llama", methods=["GET", "POST"])
def llama():
    return render_template("llama.html")


@app.route("/llama_result", methods=["POST"])
def llama_result():
    q = request.form.get("q")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ],
    )

    r = response.choices[0].message.content
    return render_template("llama_result.html", r=r)


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------
@app.route("/gemini", methods=["GET", "POST"])
def gemini():
    return render_template("gemini.html")


@app.route("/gemini_result", methods=["POST"])
def gemini_result():
    q = request.form.get("q")

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(q)
    r = response.text

    return render_template("gemini_result.html", r=r)

# ---------------------------------------------------------------------
# Paynow
# ---------------------------------------------------------------------
@app.route("/paynow", methods=["GET", "POST"])
def paynow():
    return render_template("paynow.html")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # host="0.0.0.0" is useful for Codespaces / containers
    app.run(host="0.0.0.0", port=5000, debug=True)
