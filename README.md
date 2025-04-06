🏈 Football Play Chatbot

An interactive chatbot that lets users query football play data using natural language. Built with LangChain, OpenAI (GPT-4), FAISS, and pandas, the chatbot loads structured game data from Excel files and provides intelligent answers about play stats, formations, and trends. It supports contextual follow-ups, memory, and numerical calculations (e.g., average time to throw, most successful play by gain).

✨ Features
	•	🔍 Query play data by team, formation, scheme, or play number
	•	🧠 Conversational memory using LangChain’s ConversationBufferMemory
	•	📊 Automatic calculations (averages, maximums, etc.) from structured data
	•	🧾 Works with Excel (.xlsx) game logs
	•	🧠 GPT-4 powered with retrieval-augmented generation (RAG)
	•	🔐 .env support for securely loading API keys

📦 Tech Stack
	•	Python (pandas, NumPy)
	•	LangChain for chaining and memory
	•	OpenAI API (GPT-4 Turbo)
	•	FAISS for vector similarity search
	•	dotenv for secure environment config

🚀 Getting Started
	1.	Clone this repo
	2.	Place your Excel files (e.g., montana.xlsx, idaho.xlsx) in a data/ folder
	3.	Create a .env file in the root with your OpenAI key:
    OPENAI_API_KEY=your-api-key-here
    4.	Install dependencies:
    pip install -r requirements.txt
    5.	Run the chatbot:
    python Chatbot.py
