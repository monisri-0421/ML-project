import gradio as gr
from src.predict import predict_image

def classify_image(image):
    import io
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    return predict_image(image_bytes.getvalue())

gr.Interface(fn=classify_image, inputs="image", outputs="text", title="Cat vs Dog Classifier").launch()