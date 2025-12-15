import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

device = "cuda" if torch.cuda.is_available() else "cpu"
# default: Load the model on the available device(s)
# 加载模型并移动到指定设备
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "D:\model\qwen3-vl",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model = model.to(device)

# default processer
processor = AutoProcessor.from_pretrained("D:\model\qwen3-vl")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./img/cat.png",
            },
            {"type": "text", "text": "请对这张图片进行分类,是猫还是狗"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
