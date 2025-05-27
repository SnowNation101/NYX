import json
import argparse
from tqdm import tqdm
import re
from collections import defaultdict

def extract_answer(raw_output, question_type):
    """
    Extracts the answer(s) from a given unprocessed output string based on the question type.

    Args:
        raw_output (str): The raw output string containing the answer(s) to be parsed.
        question_type (str): The type of question, which determines the parsing logic.
                             Supported types are:
                             - "single_choice": Extracts a single uppercase letter (e.g., "A").
                             - "multiple_choices": Extracts multiple uppercase letters (e.g., "ABC").
                             - "multiple_quesitons": Extracts answers for multiple questions,
                               returning all uppercase letters found.

    Returns:
        list or None: A list of extracted answers as uppercase letters.
                      Returns `None` if no match is found or if the question type is unsupported.
    """
    answer_patterns = {
        "single_choice": r"<答案>.*?([A-Z])",
        "multiple_choices": r"<答案>[^A-Z]*([A-Z]+)",
        "multiple_quesitons": r"<答案>([^<]*)"
    }

    if question_type in answer_patterns:
        if question_type == "multiple_quesitons":
            matches = re.findall(answer_patterns[question_type], raw_output)
            return [char for match in matches for char in re.findall(r"[A-Z]", match)]
        else:
            match = re.search(answer_patterns[question_type], raw_output)
            return [match.group(1)] if match else None
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaokao Bench")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--generation_path", type=str, required=True, help="The path to the output of generator")
    args = parser.parse_args()

    output_file = args.generation_path
    with open(output_file, "r", encoding='utf-8') as f:
        outputs = json.load(f)

    task = args.task
    if task == "gaokao_bench":
        from process_gaokao import get_gaokao_bench_data
        _, test = get_gaokao_bench_data()


    total = 0
    correct = 0

    # Initialize defaultdict for keyword statistics
    keyword_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, query in tqdm(enumerate(test), total=len(test)):
        correct_answer = query["answer"]
        output_answer = outputs[i]

        # Get keywords for the current question
        keywords = query["keywords"].split('_', 1)[1]
        print(keywords)


        if len(correct_answer) > 1:
            question_type = "multiple_quesitons"
        elif len(correct_answer[0]) > 1:
            question_type = "multiple_choices"
        else:
            question_type = "single_choice"
        generated_answer = extract_answer(output_answer, question_type)
        if generated_answer is None:
            generated_answer = []

        # Iterate through each sub-question and update overall and keyword statistics
        for j in range(len(correct_answer)):
            total += 1 # Increment total for overall accuracy

            is_correct = False
            if j < len(generated_answer) and generated_answer[j] == correct_answer[j]:
                correct += 1  # Increment correct for overall accuracy
                is_correct = True

            # Update keyword statistics for each keyword associated with the question
            keyword_stats[keywords]["total"] += 1
            if is_correct:
                keyword_stats[keywords]["correct"] += 1


    print(f"Total: {total}, Correct: {correct}")
    print(f"Overall Accuracy: {correct / total:.4f}")

    print("\n--- Keyword-wise Accuracy ---")
    for keyword, stats in keyword_stats.items():
        accuracy = stats["correct"] / stats["total"]
        print(f"Keyword: {keyword}, Total: {stats['total']}, Correct: {stats['correct']}, Accuracy: {accuracy:.4f}")



if __name__ == "__main__":
    main()