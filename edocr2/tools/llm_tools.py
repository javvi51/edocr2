from dotenv import load_dotenv

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

def load_llm(model_name, dimensions = None, gdt = None, tables = None):
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
