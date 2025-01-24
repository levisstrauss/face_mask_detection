import gradio as gr
from detector import MaskDetector
from pathlib import Path

# Initialize detector globally
model_path = Path('face_mask_detector_final.pth')
detector = MaskDetector(model_path)

def predict(image):
    result = detector.predict_image(image)
    is_mask = result['prediction'] == 'Mask'
    
    html_output = f"""
    <div style="width: 100%; max-width: 42rem; margin: auto; padding: 1.5rem; background: white; border-radius: 0.75rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 9999px; background: {'#22c55e' if is_mask else '#ef4444'}; margin-right: 0.5rem;"></div>
            <h2 style="font-size: 1.5rem; font-weight: bold; color: {'#15803d' if is_mask else '#b91c1c'};">
                {result['prediction']} Detected
            </h2>
        </div>
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 0.875rem; font-weight: 500; color: #374151;">Confidence Level</span>
                <span style="font-size: 0.875rem; font-weight: 500; color: #374151;">{result['confidence']:.2f}%</span>
            </div>
            <div style="width: 100%; background: #e5e7eb; border-radius: 9999px; height: 0.625rem;">
                <div style="height: 0.625rem; border-radius: 9999px; background: #2563eb; width: {result['confidence']}%; transition: width 0.5s ease;"></div>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="padding: 0.25rem 0.75rem; font-size: 0.875rem; font-weight: 500; border-radius: 9999px; 
                background: {'#dcfce7' if result['status'] == 'HIGH_CONFIDENCE' else '#fef9c3'}; 
                color: {'#166534' if result['status'] == 'HIGH_CONFIDENCE' else '#854d0e'};">
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
    title="Face Mask Detection"
)

if __name__ == "__main__":
    interface.launch()






# import gradio as gr
# from detector import MaskDetector
# from pathlib import Path

# model_path = Path('face_mask_detector_final.pth')
# detector = MaskDetector(model_path)

# def predict(image):
#     result = detector.predict_image(image)
    
#     html_output = f"""
#     <div class="w-full max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg">
#         <div class="flex items-center mb-4">
#             <div class="w-3 h-3 rounded-full {'bg-green-500' if result['prediction'] == 'Mask' else 'bg-red-500'} mr-2"></div>
#             <h2 class="text-2xl font-bold {'text-green-700' if result['prediction'] == 'Mask' else 'text-red-700'}">
#                 {result['prediction']} Detected
#             </h2>
#         </div>
        
#         <div class="mb-4">
#             <div class="flex justify-between mb-1">
#                 <span class="text-sm font-medium text-gray-700">Confidence Level</span>
#                 <span class="text-sm font-medium text-gray-700">{result['confidence']:.2f}%</span>
#             </div>
#             <div class="w-full bg-gray-200 rounded-full h-2.5">
#                 <div class="h-2.5 rounded-full bg-blue-600 transition-all duration-500" 
#                      style="width: {result['confidence']}%"></div>
#             </div>
#         </div>
        
#         <div class="flex items-center">
#             <span class="px-3 py-1 text-sm font-medium rounded-full 
#                       {'bg-green-100 text-green-800' if result['status'] == 'HIGH_CONFIDENCE' else 'bg-yellow-100 text-yellow-800'}">
#                 {result['status']}
#             </span>
#         </div>
#     </div>
#     """
    
#     return gr.HTML(html_output)

# css = """
# <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
# """

# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.HTML(),
#     title="Face Mask Detection",
#     css=css
# )

# if __name__ == "__main__":
#     interface.launch()





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

