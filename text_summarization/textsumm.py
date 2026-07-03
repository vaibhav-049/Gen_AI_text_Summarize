import gradio as gr
import re
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
	word_count = len(text.split())
	max_new_tokens = min(max(word_count // 3, 40), 120)
	min_length = min(max(word_count // 8, 20), max_new_tokens - 5)
	inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
	generated = model.generate(
		**inputs,
		max_new_tokens=max_new_tokens,
		min_length=min_length,
		num_beams=4,
		length_penalty=1.8,
		no_repeat_ngram_size=3,
		repetition_penalty=1.15,
		early_stopping=True,
		do_sample=False,
	)
	result = tokenizer.decode(generated[0], skip_special_tokens=True)
	result = re.sub(r"\s+", " ", result).strip()
	return result


with gr.Blocks() as demo:
	gr.Markdown("# Text Summarizer")
	input_text = gr.Textbox(lines=10, label="Input Text", placeholder="Paste or type text here")
	output_text = gr.Textbox(lines=10, label="Summarized Text", interactive=False)
	summarize_button = gr.Button("Summarize")
	summarize_button.click(summarize, inputs=input_text, outputs=output_text)


if __name__ == "__main__":
	demo.launch()