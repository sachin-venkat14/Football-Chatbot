import os
import json
import pandas as pd
import numpy as np
import faiss
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class FootballPlayAnalyzer:
    def __init__(self, data_files: List[str], memory_file: str = "chat_memory.json"):
        """Initialize the Football Play Analyzer with data files and configuration."""
        load_dotenv()
        self.memory_file = memory_file
        self.openai_api_key = self._get_api_key()
        self.last_game_mentioned = None
        
        # Load and process the data
        self.df = self._load_data(data_files)
        self.documents = self._create_documents()
        
        # Initialize NLP components
        self.embedding_function = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = FAISS.from_documents(documents=self.documents, embedding=self.embedding_function)
        self.llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=self.openai_api_key)
        
        # Set up memory and QA chain
        self.chat_history = self._load_memory()
        self.memory = self._initialize_memory()
        self.qa_chain = self._create_qa_chain()

    def _get_api_key(self) -> str:
        """Safely get the OpenAI API key from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("⚠️ ERROR: OPENAI_API_KEY is missing! Set it in a .env file.")
        return api_key

    def _load_data(self, files: List[str]) -> pd.DataFrame:
        """Load and process data from Excel files."""
        try:
            dataframes = []
            for file in files:
                df = pd.read_excel(file)
                game_name = os.path.basename(file).replace(".xlsx", "")
                df["Game"] = game_name
                df.rename(columns={"Clip #": "Play #", "pff_TIMETOTHROW": "Time to Throw"}, inplace=True)
                dataframes.append(df)

            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Select relevant columns
            relevant_columns = ["Game", "Play #", "R/P", "FieldPosition", "DN", "DIST", "GAIN", "Time to Throw", "FORMATION_O", "SCHEME"]
            filtered_df = combined_df[relevant_columns]
            
            # Fill missing values and convert types
            filtered_df.loc[:, "R/P"] = filtered_df["R/P"].fillna("Unknown")
            filtered_df = filtered_df.dropna(subset=["DN", "DIST"])
            
            numeric_columns = ["FieldPosition", "DN", "DIST", "GAIN", "Time to Throw"]
            filtered_df[numeric_columns] = filtered_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
            
            return filtered_df
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def _create_documents(self) -> List[Document]:
        """Create text representations for vector search."""
        self.df["Text_Representation"] = (
            "Game: " + self.df["Game"] +
            ", Play #: " + self.df["Play #"].astype(str) +
            ", Formation: " + self.df["FORMATION_O"].fillna("Unknown") +
            ", Scheme: " + self.df["SCHEME"].fillna("Unknown") +
            ", Play type: " + self.df["R/P"]
        )
        
        return [
            Document(
                page_content=row["Text_Representation"],
                metadata={"Play #": row["Play #"], "Game": row["Game"]}
            )
            for _, row in self.df.iterrows()
        ]

    def _load_memory(self) -> List[Dict[str, str]]:
        """Load previous chat memory if available."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print(f"Error loading memory file. Starting with fresh memory.")
                return []
        return []

    def _save_memory(self) -> None:
        """Save chat memory to file."""
        MAX_HISTORY_LENGTH = 50
        if len(self.chat_history) > MAX_HISTORY_LENGTH:
            self.chat_history = self.chat_history[-MAX_HISTORY_LENGTH:]
            
        try:
            with open(self.memory_file, "w") as file:
                json.dump(self.chat_history, file)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def _initialize_memory(self) -> ConversationBufferMemory:
        """Initialize memory from saved chat history."""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in self.chat_history:
            memory.chat_memory.add_user_message(msg["user"])
            memory.chat_memory.add_ai_message(msg["bot"])
        return memory

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create the conversational retrieval chain."""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 100}),
            memory=self.memory
        )

    def process_query(self, query: str) -> str:
        """Process user query and generate response."""
        if not query or query.strip() == "":
            return "Error: No query provided."

        # Detecting game name
        game_name = None
        for game in self.df["Game"].unique():
            if game.lower() in query.lower():
                game_name = game
                self.last_game_mentioned = game
                break

        if "most successful play" in query.lower():
            if self.df["GAIN"].dropna().empty:
                return "No valid gain data available to determine the most successful play."
            most_successful_play = self.df.loc[self.df["GAIN"].idxmax()]
            return (
                f"The most successful play was Play {most_successful_play['Play #']} in the {most_successful_play['Game']} game, "
                f"which gained {most_successful_play['GAIN']} yards."
            )

        if "time to throw" in query.lower():
            if game_name:
                game_values = self.df[self.df["Game"] == game_name]["Time to Throw"].dropna()
                if game_values.empty:
                    return f"No valid time to throw data found for the {game_name} game."
                avg_time = game_values.mean()
            else:
                valid_values = self.df["Time to Throw"].dropna()
                if valid_values.empty:
                    return "No valid time to throw data found across all games."
                avg_time = valid_values.mean()

            return f"The average time to throw {'in the ' + game_name + ' game' if game_name else 'across all games'} is {avg_time:.2f} seconds."

        try:
            response = self.qa_chain.invoke({"question": query})
            answer = response["answer"]
            self.chat_history.append({"user": query, "bot": answer})
            self._save_memory()
            return answer
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"

# Example usage
if __name__ == "__main__":
    analyzer = FootballPlayAnalyzer(
        data_files=["/Users/sachinvenkat/Downloads/montana.xlsx", "/Users/sachinvenkat/Downloads/idaho.xlsx"],
        memory_file="chat_memory.json"
    )
    
    print("Football Play Analyzer is running. Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        bot_response = analyzer.process_query(user_query)
        print("Bot:", bot_response)