import gradio as gr
from detector import MaskDetector
from pathlib import Path

model_path = Path('models/face_mask_detector_final.pth')
detector = MaskDetector(model_path)

def predict(image):
    result = detector.predict_image(image)
    
    html_output = f"""
    <div class="w-full max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg">
        <div class="flex items-center mb-4">
            <div class="w-3 h-3 rounded-full {'bg-green-500' if result['prediction'] == 'Mask' else 'bg-red-500'} mr-2"></div>
            <h2 class="text-2xl font-bold {'text-green-700' if result['prediction'] == 'Mask' else 'text-red-700'}">
                {result['prediction']} Detected
            </h2>
        </div>
        
        <div class="mb-4">
            <div class="flex justify-between mb-1">
                <span class="text-sm font-medium text-gray-700">Confidence Level</span>
                <span class="text-sm font-medium text-gray-700">{result['confidence']:.2f}%</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2.5">
                <div class="h-2.5 rounded-full bg-blue-600 transition-all duration-500" 
                     style="width: {result['confidence']}%"></div>
            </div>
        </div>
        
        <div class="flex items-center">
            <span class="px-3 py-1 text-sm font-medium rounded-full 
                      {'bg-green-100 text-green-800' if result['status'] == 'HIGH_CONFIDENCE' else 'bg-yellow-100 text-yellow-800'}">
                {result['status']}
            </span>
        </div>
    </div>
    """
    
    return gr.HTML(html_output)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.HTML(),
    title="Face Mask Detection",
    css="tailwind"
)

if __name__ == "__main__":
    interface.launch()





# import gradio as gr
# from detector import MaskDetector
# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize detector
# model_path = Path('face_mask_detector_final.pth')
# detector = MaskDetector(model_path)

# def predict(image):
#     try:
#         result = detector.predict_image(image)
#         return f"{result['prediction']} (Confidence: {result['confidence']:.2f}%, Status: {result['status']})"
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return "Error processing image"

# # Create Gradio interface
# iface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Text(),
#     title="Face Mask Detection",
#     description="Upload an image to detect if a person is wearing a face mask.",
#     examples=[
#         ["examples/9.png"],
#         ["examples/28.png"]
#     ],
#     cache_examples=True
# )

# # Launch app
# if __name__ == "__main__":
#     iface.launch()

