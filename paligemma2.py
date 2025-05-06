from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

model_id = "google/paligemma2-10b-ft-docci-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
model = model.to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

prompt = "<image>caption en"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
raw_image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")

inputs = processor(prompt, raw_image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)

input_len = inputs["input_ids"].shape[-1]
print(processor.decode(output[0][input_len:], skip_special_tokens=True))