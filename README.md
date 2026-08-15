# AI Chatbot (Python, NLP + Machine Learning)

A small intent-based chatbot. It classifies each user message into an
**intent** (greeting, goodbye, hours, joke, etc.) using TF-IDF + Logistic
Regression, then replies with a matching response.

## Files

- `intents.json` — training data: example phrases ("patterns") per intent,
  and the possible responses for each intent.
- `chatbot.py` — preprocessing, model training, and the chat loop.
- `chatbot_model.pkl` — saved trained model (created automatically the first
  time you run the bot).

## Setup

```bash
pip install -r requirements.txt
```

No extra downloads (like NLTK's `punkt`) are needed — text cleaning is done
with plain regex, so it works offline out of the box.

## Run

```bash
python chatbot.py
```

Type messages at the `You:` prompt; type `quit` or `exit` to stop.

To force the model to retrain (e.g. after editing `intents.json`):

```bash
python chatbot.py --retrain
```

## How it works

1. **NLP step** — each message is lowercased, stripped of punctuation, and
   lightly stemmed (`preprocess()` in `chatbot.py`).
2. **ML step** — cleaned text is turned into TF-IDF vectors and classified
   with a `LogisticRegression` model trained on the patterns in
   `intents.json`.
3. **Confidence check** — if the model isn't confident in its top prediction
   (below `CONFIDENCE_THRESHOLD` in `chatbot.py`), the bot uses a `fallback`
   response instead of guessing.
4. A response is chosen randomly from the predicted intent's response list.

## Extending it

- **Add new topics**: open `intents.json` and add a new object with a `tag`,
  a handful of example `patterns`, and one or more `responses`. Add at least
  5-10 varied patterns per intent for decent accuracy. Then run with
  `--retrain`.
- **Improve accuracy**: more/varied training patterns per intent helps the
  most. You can also swap `LogisticRegression` for `LinearSVC` or a
  `RandomForestClassifier` in `chatbot.py` — same interface.
- **Add real NLP**: swap the hand-rolled `simple_stem()` for NLTK's
  `WordNetLemmatizer` or spaCy's lemmatizer if you have those installed and
  want more accurate normalization.
- **Add memory/context**: currently every message is classified
  independently. For multi-turn context (e.g. "book a table" → "for how many
  people?"), you'd track a `current_intent`/slot-filling state between turns.
