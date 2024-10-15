from dotenv import load_dotenv
import numpy as np
from PIL import Image
import ast

def ask_llm(model, tokenizer, system_role, prompt):
    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def correct_dimensions(model, tokenizer, dimensions):
    common_errors = '''Tolerances: If present, ensure tolerances appear after the nominal value.
    Possible error: '⌀' confused with '0'.
    Posible error: missing decimal points ('.' or ',')'''

    examples = '''input: 05; output: ⌀5
    input: 10, +02, -0; output: 10, +0.2, -0
    input: 40; output: 40'''

    system_role = f'''You are a specialized system for correcting OCR output from mechanical drawings. 
    Given a list of OCR-extracted values, separated by ';' return the corrected values list.
    Only the list is expected as output.
    These are some common errors:\n{common_errors}\nExamples:\n{examples}'''

    processed_string = '; '.join([f"{item[0]}" for item in dimensions])
    prompt = processed_string

    response = ask_llm(model, tokenizer, system_role, prompt)
    return response

def load_llm_(model_name, dimensions = None, gdt = None, tables = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    load_dotenv()
    
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dimensions:
        elements = correct_dimensions(model, tokenizer, dimensions)
        # Regular expression to match the pattern of 'text(coordinate1,coordinate2)'
        elements = elements.split(';')   # Split by ';'

        # Strip whitespace and convert to appropriate types
        result_list = [elem.strip() for elem in elements]
        llm_dimensions = []
        for el in range(len(result_list)):
            llm_dimensions.append([result_list[el].strip(), (dimensions[el][0],dimensions[el][1])])

        return llm_dimensions

def query(model, index, embeddings, query_text):
    
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model)

    # Function to normalize an embedding (for cosine similarity)
    def normalize(embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm

    # Generate the query embedding
    query_embedding = model.encode(query_text)

    # Normalize the query embedding
    query_embedding = normalize(query_embedding)

    # Perform the search - retrieving top 5 most similar results
    k = 5  # Number of results you want
    query_embedding = query_embedding.reshape(1, -1)  # Reshape to match FAISS input
    distances, indices = index.search(query_embedding, k)

    # Map the retrieved indices to drawing names
    retrieved_drawings = [list(embeddings.keys())[idx] for idx in indices[0]]

    # Print or return the retrieved drawing names
    print("Most relevant drawings:", retrieved_drawings)

def call_vision_infoblock(img, query, model, processor, device):
    
    from qwen_vl_utils import process_vision_info
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    query_string = ', '.join(query)

    messages = [
        {"role": "user",
            "content": [{"type": "image","image": img,},
                        {"type": "text", "text": f"Based on the image, return only a python dictionary extracting this information: {query_string}"},],
        }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    cleaned_output = output_text[0].strip('```python\n```')
    # Convert the cleaned string into a dictionary
    return ast.literal_eval(cleaned_output)

def load_llm(model_name = "Qwen/Qwen2-VL-7B-Instruct"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto")
    
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor
