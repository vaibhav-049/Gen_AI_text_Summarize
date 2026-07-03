import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
model.to(device)


def summarize(text):
	if not text.strip():
		return ""
	inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
	generated = model.generate(**inputs, max_new_tokens=130, min_length=30, do_sample=False)
	return tokenizer.decode(generated[0], skip_special_tokens=True)


with gr.Blocks() as demo:
	gr.Markdown("# Text Summarizer")
	input_text = gr.Textbox(lines=10, label="Input Text", placeholder="Paste or type text here")
	output_text = gr.Textbox(lines=10, label="Summarized Text", interactive=False)
	summarize_button = gr.Button("Summarize")
	summarize_button.click(summarize, inputs=input_text, outputs=output_text)


if __name__ == "__main__":
	demo.launch()