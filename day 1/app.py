from flask import Flask, render_template, jsonify
import requests
import random

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/kural/<int:number>")
def get_kural(number):

    if number < 1 or number > 1330:
        return jsonify({
            "error": "Kural number must be between 1 and 1330"
        }), 400

    url = f"https://tamil-kural-api.vercel.app/api/kural/{number}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return jsonify({
                "error": "Unable to get Kural from API"
            }), 500

        data = response.json()

        raw_kural = data.get("kural", "")
        line1, line2 = "", ""

        if isinstance(raw_kural, list):
            if len(raw_kural) >= 2:
                line1 = raw_kural[0]
                line2 = raw_kural[1]
            elif len(raw_kural) == 1:
                line1 = raw_kural[0]
        elif isinstance(raw_kural, str):
            parts = [p.strip() for p in raw_kural.split("\n") if p.strip()]
            if len(parts) >= 2:
                line1 = parts[0]
                line2 = parts[1]
            else:
                line1 = raw_kural

        raw_meaning = data.get("meaning", "")

        tamil_meaning = ""
        english_meaning = ""

        if isinstance(raw_meaning, dict):
            # Pick best Tamil meaning
            tamil_meaning = (
                raw_meaning.get("ta_mu_va")
                or raw_meaning.get("ta_salamon")
                or raw_meaning.get("ta_kalaignar")
                or ""
            )
            english_meaning = raw_meaning.get("en", "")
        elif isinstance(raw_meaning, str):
            tamil_meaning = raw_meaning

        # Legacy 'porul' field fallback
        if not tamil_meaning:
            porul = data.get("porul", "")
            if isinstance(porul, dict):
                tamil_meaning = porul.get("tam") or porul.get("tamil") or ""
            elif isinstance(porul, str):
                tamil_meaning = porul

        return jsonify({
            "number": number,
            "line1": line1,
            "line2": line2,
            "tamil_meaning": tamil_meaning,
            "english_meaning": english_meaning,
            "chapter": data.get("chapter", ""),
            "section": data.get("section", "")
        })

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "API connection failed"
        }), 500


@app.route("/api/random")
def get_random_kural():
    random_number = random.randint(1, 1330)
    return get_kural(random_number)


if __name__ == "__main__":
    app.run(debug=True)
