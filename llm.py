import os
from dotenv import load_dotenv
import google.generativeai as genai

def generate_jd(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return {"data": response.parts[0].text}
    except Exception as e:
        return f"API error: {str(e)}"

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    if not api_key:
        print("API key not found in .env file")
        return

    print(generate_jd("Generate a job description for a software engineer"))

if __name__ == "__main__":
    main()