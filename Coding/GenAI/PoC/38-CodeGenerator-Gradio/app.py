from gradio.sketch.run import create

demo = create("test.py", "test.py.json")
demo.launch()