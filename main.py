from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from bson import ObjectId
import time
from datetime import datetime

# AI Configuration
generate_llm = ChatGroq(
    temperature=0.3,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    max_tokens=8000,
    timeout=60
)

def generate_response(prompt_template: str, max_tokens: int = 8000, retries=3, delay=1, max_wait_time=60):
    total_wait_time = 0
    prompt = HumanMessage(content=prompt_template)
    for attempt in range(retries):
        if total_wait_time >= max_wait_time:
            print("Total wait time exceeded. Stopping retries.")
            return None

        try:
            response = generate_llm.invoke(prompt.content)
            return response.content
        except Exception as e:
            print(f"Exception occurred: {e}")
            if attempt < retries - 1:
                backoff_time = delay * (2 ** attempt)  # Exponential backoff
                total_wait_time += backoff_time
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                print("Max retries exceeded. Please check your settings.")
                return None

# FastAPI Configuration
app = FastAPI()

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["MoodTrackerDB"]
collection = db["moodResponses"]

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Body Model
class MoodInput(BaseModel):
    userId: str
    mood: str
    intensity: int

# Helper Function: Generate Sentiment and Guidance
def generate_sentiment_guidance_and_aspects(mood: str, intensity: int) -> dict:
    """
    Generates sentiment, guidance, and key aspects based on the user's mood and intensity.

    Args:
        mood (str): The user's mood description.
        intensity (int): The intensity of the mood.

    Returns:
        dict: A dictionary containing 'sentiment', 'guidance', and 'aspects'.
    """
    # Generate prompt for AI to analyze sentiment, guidance, and aspects
    prompt = (
        f"You are a friendly AI assistant."
        f"A user described their mood as: '{mood}' with an intensity of {intensity}. "
        f"Analyze this input and provide:\n"
        f"1. Sentiment in one word only on the first line (e.g., 'positive', 'negative', 'neutral', 'slightly positive', or 'slightly negative').\n"
        f"2. Motivational guidance on the second line, explaining how the user can improve or maintain their state of mind. Convince user it is easy to overcome\n"
        f"3. Key aspects of the mood (e.g., 'work', 'relationships', 'health') on the third line, separated by commas.\n"
        f"Do not include any extra text or formatting. Ensure the output is exactly three lines."
    )

    # AI Response
    ai_response = generate_response(prompt)
    print(f"AI Response:\n{ai_response}")
    if not ai_response:
        raise HTTPException(status_code=500, detail="AI sentiment, guidance, and aspects generation failed.")

    # Parse the AI response
    try:
        # Split the response into lines
        lines = ai_response.strip().split("\n")
        if len(lines) < 3:
            raise ValueError("Incomplete AI response. Expecting exactly three lines: sentiment, guidance, and aspects.")

        # Extract sentiment, guidance, and aspects
        sentiment = lines[0].strip()  # First line is sentiment
        guidance = lines[1].strip()  # Second line is guidance
        aspects = [aspect.strip() for aspect in lines[2].split(",")]  # Third line is aspects, split by commas

        return {"sentiment": sentiment, "guidance": guidance, "aspects": aspects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")

@app.post("/analyze/")
async def analyze(input: MoodInput):
    # Generate sentiment, guidance, and aspects
    result = generate_sentiment_guidance_and_aspects(input.mood, input.intensity)

    # Add date and time to the document
    now = datetime.now()
    document = {
        "userId": input.userId,
        "mood": input.mood,
        "intensity": input.intensity,
        "sentiment": result["sentiment"],
        "guidance": result["guidance"],
        "aspects": result["aspects"],
        "date": now.strftime("%Y-%m-%d"),  # Date in YYYY-MM-DD format
        "time": now.strftime("%H:%M:%S")   # Time in HH:MM:SS format
    }

    # Save to MongoDB
    db_result = collection.insert_one(document)

    # Include the ObjectId as a string in the response
    response = {**document, "_id": str(db_result.inserted_id)}
    return response


@app.get("/mood-logs/")
async def get_mood_logs(userId: str = Query(...)):
    """
    Fetch mood logs for a specific user by userId.
    """
    try:
        # Query logs for the specific userId
        logs = list(collection.find({"userId": userId}))
        # Convert ObjectId to string
        for log in logs:
            log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching mood logs: {str(e)}")
    


class RecoveryInput(BaseModel):
    logId: str
    userId: str
    recovery: str

@app.post("/recovery-submit/")
async def submit_recovery(input: RecoveryInput):
    """
    Endpoint to handle recovery submissions for logs.
    """
    try:
        # Convert logId to ObjectId
        log_id = ObjectId(input.logId)

        # Ensure the log exists and belongs to the user
        log = collection.find_one({"_id": log_id, "userId": input.userId})
        if not log:
            raise HTTPException(status_code=404, detail="Log not found or unauthorized access.")

        # Add the recovery input to the log
        result = collection.update_one(
            {"_id": log_id},
            {"$set": {"recovery": input.recovery}}
        )

        # Confirm the update
        if result.matched_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update recovery input.")

        return {"message": "Recovery input submitted successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")