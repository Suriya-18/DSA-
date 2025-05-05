def char_to_number(ch):
    if ch == ' ':
        return 0
    elif ch == ',':
        return 99
    elif ch == '.':
        return 100
    elif 'a' <= ch <= 'z':
        return ord(ch) - ord('a') + 1
    elif 'A' <= ch <= 'Z':
        return ord(ch) - ord('A') + 27
    else:
        return None

def initial_conversion(text):
    return [char_to_number(ch) for ch in text if char_to_number(ch) is not None]

def process_words(text, initial_nums):
    words = text.split(' ')
    result = []
    idx = 0
    for word in words:
        length = len([ch for ch in word if ch.isalpha()])
        processed_word = []
        for ch in word:
            num = char_to_number(ch)
            if num in [99, 100]:  # punctuation
                processed_word.append(num)
            elif ch == ' ':
                processed_word.append(0)
            elif ch.isalpha():
                if length % 2 == 0:
                    processed_word.append(num - 10)
                else:
                    processed_word.append(num + 50)
        result.extend(processed_word)
        result.append(0)  # space after word
    result = result[:-1]  # remove last extra space
    return result

def final_adjustment(nums):
    total = sum(nums)
    if total % 2 == 0:
        return [n + 5 for n in nums], total
    else:
        return [n - 3 for n in nums], total

def print_output(title, data):
    print(f"{title}\n{'='*len(title)}")
    print(data)
    print(f"Sum: {sum(data)}\n")

if __name__ == "__main__":
    text = "Success is not final, failure is not fatal. It is the courage to continue that counts."
    text = text.strip()

    # Step 1: Initial Conversion
    step1 = initial_conversion(text)
    print_output("1. Numeric sequence after initial conversion", step1)

    # Step 2: Word Processing
    step2 = process_words(text, step1)
    print_output("2. Sequence after word processing", step2)

    # Step 3: Final Adjustment
    step3, sum2 = final_adjustment(step2)
    print_output("3. Final encrypted output", step3)
