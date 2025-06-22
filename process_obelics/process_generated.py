import json
import re

# Path to the input JSON file
input_path = "process_obelics/processed_qa.json"
# Path to the output JSON file
output_path = "process_obelics/qa_flattened.json"

# Load the original data
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Store the extracted QA pairs
flattened_data = []

for item in data:
    pos_text = item["doc"]
    pos_image_path = item["images"]
    qa_block = item["generated_qa"]

    # Extract questions and answers using regex
    q_list = re.findall(r"\[Q\d+\]: (.*?)\n", qa_block)
    a_list = re.findall(r"\[A\d+\]: (.*?)(?=\n\[Q\d+\]:|\Z)", qa_block, flags=re.S)

    # Trim whitespace
    q_list = [q.strip() for q in q_list]
    a_list = [a.strip() for a in a_list]

    for q, a in zip(q_list, a_list):
        # Find all <imageN> tags
        image_refs = re.findall(r"<image(\d+)>", q)
        # Get corresponding image paths
        qry_image_path = []
        for ref in image_refs:
            idx = int(ref) - 1  # Convert to 0-based index
            if 0 <= idx < len(pos_image_path):
                qry_image_path.append(pos_image_path[idx])
        # Remove duplicates in case of multiple same <imageN>
        qry_image_path = list(dict.fromkeys(qry_image_path))

        # Replace <imageN> with <|image|>
        q = re.sub(r"<image\d+>", "<|image|>", q)

        flattened_data.append({
            "qry": q,
            "ans": a,
            "pos_text": pos_text,
            "pos_image_path": pos_image_path,
            "qry_image_path": qry_image_path
        })

# Filter out items where the count of "<|image|>" does not match the number of image paths
flattened_data = [
    item for item in flattened_data
    if item["qry"].count("<|image|>") == len(item["qry_image_path"])
]

# Save the result
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(flattened_data, f, ensure_ascii=True, indent=2)

print(f"Extraction complete. {len(flattened_data)} QA pairs saved to {output_path}")
