import random

def ask_question(level=1):
    if level == 1:
        a, b = random.randint(1, 10), random.randint(1, 10)
        answer = a + b
        question = f"What is {a} + {b}? "
    elif level == 2:
        a, b = random.randint(5, 20), random.randint(1, 10)
        answer = a - b
        question = f"What is {a} - {b}? "
    else:
        a, b = random.randint(2, 10), random.randint(2, 10)
        answer = a * b
        question = f"What is {a} × {b}? "
    return question, answer

def run_agent():
    level = 1
    score = 0

    print("👋 Hello! Let’s play a math game.")
    for i in range(5):  # ask 5 questions
        q, ans = ask_question(level)
        try:
            user_ans = int(input(q))
            if user_ans == ans:
                print("🎉 Correct! Great job!")
                score += 1
                if score >= 3:  # level up after 3 correct answers
                    level = min(level + 1, 3)
            else:
                print(f"❌ Oops! The right answer was {ans}. Keep trying!")
        except ValueError:
            print("Please enter a number.")

    print(f"Game over! ⭐ You scored {score}/5.")

if __name__ == "__main__":
    run_agent()
