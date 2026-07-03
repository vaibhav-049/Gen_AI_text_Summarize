# Text Summarization Web App

Minimal Gradio app with two text boxes: one for input, one for the summarized result.

## Run Locally

```bash
python -m pip install -r requirements.txt
python app.py
```

## Publish to Hugging Face Spaces

1. Create a new Space on Hugging Face and choose Gradio.
2. Upload or push these files: `app.py`, `textsumm.py`, `requirements.txt`, `README.md`.
3. Wait for the Space build to finish. The app becomes live automatically.

## Hugging Face Login

```bash
pip install huggingface_hub
huggingface-cli login
```

## If you want to push from git

```bash
git clone https://huggingface.co/spaces/<username>/<repo>
cd <repo>
git add .
git commit -m "Initial commit"
git push
```

If the model is private, add `HF_TOKEN` in Space settings.

## Push to GitHub First

```bash
git init
git add .
git commit -m "Add text summarizer app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

After that, you can import the GitHub repo into Hugging Face Spaces or push the same files into the Space repo.
