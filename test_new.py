import json
from tqdm import tqdm
from collections import defaultdict
from src.utils.process_gaokao import get_gaokao_mm_data
import re

output_file = "outputs/generations/generated_250529_194608.json"

def extract_answer(output):
    matches = re.findall(r"答案.*?([A-D])", output, re.DOTALL)
    return ''.join(matches) if matches else None

def main():
    with open(output_file, "r", encoding='utf-8') as f:
        outputs = json.load(f)

    _, mm_test = get_gaokao_mm_data()

    keyword_overall_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    keyword_first_answer_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_attempt_distribution = defaultdict(int)

    for i, query in tqdm(enumerate(mm_test), total=len(mm_test)):
        keyword = query["keywords"]
        correct_answer = query["answer"][0]

        # Statistics for overall accuracy by keyword (if any output is correct)
        keyword_overall_stats[keyword]["total"] += 1
        found_correct_in_any_output = False
        
        # New: Track at which attempt the answer was correct
        attempt_number = 0 
        
        for j in range(len(outputs[i])):
            output_answer = outputs[i][j]
            predicted_answer = extract_answer(output_answer)
             
            if correct_answer == predicted_answer:
                keyword_overall_stats[keyword]["correct"] += 1
                found_correct_in_any_output = True
                attempt_number = j + 1 # +1 because j is 0-indexed
                break
        
        # New: Update the distribution of correct attempts
        correct_attempt_distribution[attempt_number] += 1


        # Statistics for the first answer's accuracy by keyword
        keyword_first_answer_stats[keyword]["total"] += 1
        if outputs[i]:  # Check if there's at least one output
            first_output_answer = outputs[i][0]
            predicted_first_answer = extract_answer(first_output_answer)
            if correct_answer == predicted_first_answer:
                keyword_first_answer_stats[keyword]["correct"] += 1

    # Calculate and print overall accuracy by keyword (considering any correct output)
    keyword_overall_acc = {}
    for keyword, stats in keyword_overall_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        keyword_overall_acc[keyword] = {
            "total_questions": stats["total"],
            "correct_answers": stats["correct"],
            "accuracy": acc
        }

    total_correct_overall = sum(stats["correct"] for stats in keyword_overall_stats.values())
    total_all_overall = sum(stats["total"] for stats in keyword_overall_stats.values())
    print(f"Overall Acc (considering any correct output): {total_correct_overall / total_all_overall:.4f}")

    print("\n---")
    print("Accuracy by Keyword (considering any correct output):")
    for keyword, stats in sorted(keyword_overall_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"{keyword}:")
        print(f"  - Total Questions: {stats['total_questions']}")
        print(f"  - Correct Answers: {stats['correct_answers']}")
        print(f"  - Accuracy: {stats['accuracy']:.4f}\n")

    # Calculate and print accuracy for the first answer by keyword
    keyword_first_answer_acc = {}
    for keyword, stats in keyword_first_answer_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        keyword_first_answer_acc[keyword] = {
            "total_questions": stats["total"],
            "correct_answers": stats["correct"],
            "accuracy": acc
        }
    
    total_correct_first_answer = sum(stats["correct"] for stats in keyword_first_answer_stats.values())
    total_all_first_answer = sum(stats["total"] for stats in keyword_first_answer_stats.values())
    print(f"\n---")
    print(f"Overall Acc (first answer for each question): {total_correct_first_answer / total_all_first_answer:.4f}")

    print("\nAccuracy of the first answer by Keyword:")
    for keyword, stats in sorted(keyword_first_answer_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"{keyword}:")
        print(f"  - Total Questions: {stats['total_questions']}")
        print(f"  - Correct First Answers: {stats['correct_answers']}")
        print(f"  - Accuracy: {stats['accuracy']:.4f}\n")

    # New: Print the distribution of correct attempts
    print("\n---")
    print("Distribution of Correct Answer Attempts:")
    for attempt, count in sorted(correct_attempt_distribution.items()):
        if attempt == 0:
            print(f"  - Questions with no correct answer among outputs: {count}")
        else:
            print(f"  - Correct at attempt #{attempt}: {count}")


if __name__ == "__main__":
    main()