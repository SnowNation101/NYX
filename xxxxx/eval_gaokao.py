import json
from tqdm import tqdm
from collections import defaultdict
from process_gaokao import get_gaokao_mm_data
from process_gaokao import get_gaokao_bench_data
import re

def extract_answer(output):
    matches = re.findall(r"答案.*?([A-D])", output, re.DOTALL)
    return ''.join(matches) if matches else None

output_file = "outputs_utf8.json"
with open(output_file, "r", encoding='utf-8') as f:
    outputs = json.load(f)

_, mm_test = get_gaokao_mm_data()

keyword_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for i, query in tqdm(enumerate(mm_test), total=len(mm_test)):
    correct_answer = query["answer"][0]
    output_answer = outputs[i][0]
    predicted_answer = extract_answer(output_answer)
    
    keyword = query["keywords"]
    
    keyword_stats[keyword]["total"] += 1
    if correct_answer == predicted_answer:
        keyword_stats[keyword]["correct"] += 1

keyword_acc = {}
for keyword, stats in keyword_stats.items():
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    keyword_acc[keyword] = {
        "total_questions": stats["total"],
        "correct_answers": stats["correct"],
        "accuracy": acc
    }

total_correct = sum(stats["correct"] for stats in keyword_stats.values())
total_all = sum(stats["total"] for stats in keyword_stats.values())
print(f"Overall Acc: {total_correct / total_all:.4f}")

print("\nAccuracy by Keyword:")
for keyword, stats in sorted(keyword_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    print(f"{keyword}:")
    print(f"  - Total Questions: {stats['total_questions']}")
    print(f"  - Correct Answers: {stats['correct_answers']}")
    print(f"  - Accuracy: {stats['accuracy']:.4f}\n")