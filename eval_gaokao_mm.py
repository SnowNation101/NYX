import json
from tqdm import tqdm
from collections import defaultdict
from src.utils.process_gaokao import get_gaokao_mm_data
from src.utils.process_gaokao import get_gaokao_bench_data
import re

def extract_answer(output):
    matches = re.findall(r"答案.*?([A-D])", output, re.DOTALL)
    return ''.join(matches) if matches else None



def cal_score(output_file):
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



def get_reward_dataset(output_file, retrieved_docs_file):
    with open(output_file, "r", encoding='utf-8') as f:
        outputs = json.load(f)

    with open(retrieved_docs_file, "r", encoding='utf-8') as f:
        retrieved_docs = json.load(f)

    _, mm_test = get_gaokao_mm_data()
    reward_dataset = []

    for i, query in tqdm(enumerate(mm_test), total=len(mm_test)):
        correct_answer = query["answer"][0]
        answers = outputs[i]
        docs = retrieved_docs[i]['retrieved']

        # Find the index of the first correct answer
        pos_idx = next(
            (j for j, ans in enumerate(answers) if extract_answer(ans) == correct_answer),
            0  # Default to the first answer if none match
        )

        entry = {
            "query": query,
            "pos": docs[pos_idx],
            "neg": [doc for j, doc in enumerate(docs) if j != pos_idx]
        }

        new_entry = {
            "qry_img": entry['query']['picture'],
            "qry_txt": "请检索最相关的高考题。" + entry['query']['question'],
            "pos_img": entry['pos']['picture'],
            "pos_txt": entry['pos']['text'],
            "neg_img": [doc['picture'] for doc in entry['neg']],
            "neg_txt": [doc['text'] for doc in entry['neg']],
        }


        reward_dataset.append(new_entry)

    return reward_dataset


def main():
    output_file = "outputs/generations/generated_250529_194608.json"
    retrieved_docs_file = "outputs/retrievals/retrieved_gaokao_mm_0527.json"
    # cal_score(output_file)
    reward_dataset = get_reward_dataset(output_file, retrieved_docs_file)
    
    with open("outputs/preference_data/pref_gaokao_mm_0610.json", "w", encoding='utf-8') as f:
        json.dump(reward_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()