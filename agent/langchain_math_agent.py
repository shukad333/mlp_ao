from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI   # or replace with local model
# For local: from langchain_community.chat_models import ChatOllama

app = FastAPI()

# request/response models
class MathRequest(BaseModel):
    question: str
    user_answer: int
    correct_answer: int

class MathResponse(BaseModel):
    is_correct: bool
    explanation: str

# init LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# For free/local: llm = ChatOllama(model="llama3")

# prompt template
prompt = PromptTemplate(
    input_variables=["question", "user_answer", "correct_answer"],
    template="""
You are a friendly math tutor for a 6-year-old child.
Question: {question}
Child's answer: {user_answer}
Correct answer: {correct_answer}

If the child is correct:
 - Praise them with fun and encouragement.
If the child is wrong:
 - Gently explain why it's wrong in very simple words.
 - Show a quick trick or example to help them learn.

Keep it short, playful, and motivating.
"""
)

@app.post("/check", response_model=MathResponse)
async def check_answer(req: MathRequest):
    is_correct = req.user_answer == req.correct_answer
    explanation = llm.invoke(
        prompt.format(
            question=req.question,
            user_answer=req.user_answer,
            correct_answer=req.correct_answer
        )
    ).content
    return MathResponse(is_correct=is_correct, explanation=explanation)
