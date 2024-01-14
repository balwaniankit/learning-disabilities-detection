import pandas as pd

def evaluate_quiz(questions_df, user_answers):
    correct_answers = questions_df['Answer']
    score = sum(user_answer == correct_answer for user_answer, correct_answer in zip(user_answers, correct_answers))
    total_questions = len(questions_df)
    return score, total_questions

#     # total_questions = len(questions_df)
#     message = "Thanks for submitting!!"
#     return message
