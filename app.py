import gradio as gr
from transformers import pipeline

# Load different pipelines
sentiment_pipe = pipeline("sentiment-analysis")
summary_pipe = pipeline("summarization")
translation_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
qa_pipe = pipeline("question-answering")
zeroshot_pipe = pipeline("zero-shot-classification")

def process_text(task, text, question=None, candidate_labels=None):
    if not text.strip():
        return "Please enter some text."

    if task == "Sentiment Analysis":
        return sentiment_pipe(text)[0]
    
    elif task == "Summarization":
        return summary_pipe(text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    
    elif task == "Translation (FR→EN)":
        return translation_pipe(text)[0]["translation_text"]
    
    elif task == "Question Answering":
        if not question:
            return "Please enter a question."
        return qa_pipe(question=question, context=text)["answer"]
    
    elif task == "Zero-Shot Classification":
        if not candidate_labels:
            return "Please provide comma-separated labels."
        labels = [label.strip() for label in candidate_labels.split(",")]
        return zeroshot_pipe(text, candidate_labels=labels)

# Gradio UI
with gr.Blocks(title="Text Intelligence Playground") as demo:
    gr.Markdown("## 🧠 Text Intelligence Playground\nChoose a text task and explore various NLP capabilities.")

    task = gr.Dropdown(
        ["Sentiment Analysis", "Summarization", "Translation (FR→EN)", "Question Answering", "Zero-Shot Classification"],
        label="Select Task",
        value="Sentiment Analysis"
    )
    text = gr.Textbox(lines=6, label="Input Text")
    question = gr.Textbox(label="Question (for QA only)", visible=False)
    labels = gr.Textbox(label="Candidate Labels (for Zero-Shot, comma separated)", visible=False)
    output = gr.JSON(label="Output")

    def update_inputs(selected):
        return (
            gr.update(visible=selected == "Question Answering"),
            gr.update(visible=selected == "Zero-Shot Classification")
        )

    task.change(update_inputs, task, [question, labels])
    run_btn = gr.Button("Run")
    run_btn.click(process_text, [task, text, question, labels], output)

demo.launch()
