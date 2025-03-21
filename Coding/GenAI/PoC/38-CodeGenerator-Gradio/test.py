import gradio as gr

with gr.Blocks() as demo:
    audio = gr.Audio(type="filepath")
    with gr.Row():
        textbox = gr.Textbox()
        markdown = gr.Markdown()
    button = gr.Button()

demo.launch()