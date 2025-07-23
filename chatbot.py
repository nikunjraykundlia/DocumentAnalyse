import logging

class Chatbot:
    def __init__(self, ollama_client):
        """
        Initializes the Chatbot with an Ollama client.
        Args:
            ollama_client: An initialized Ollama client instance.
        """
        self.ollama_client = ollama_client
        logging.info("Chatbot initialized.")

    def answer_question(self, question: str, context: str) -> str:
        """
        Answers a user's question based on the provided document context using Ollama.
        Args:
            question: The user's question.
            context: The text context extracted from the uploaded documents.
        Returns:
            The AI-generated answer.
        """
        if not self.ollama_client:
            return "I can't answer questions right now as the chat functionality is not available."

        # Stricter prompt to ensure answers are based only on the provided context.
        system_prompt = """
        You are 'DocumentChat', a professional AI assistant. Your ONLY function is to answer questions based on the text provided to you.
        You must follow these rules strictly:
        1. Your entire knowledge base is the 'DOCUMENT CONTEXT' provided below. Do not use any external or pre-trained knowledge.
        2. Answer the user's 'QUESTION' using ONLY the information found in the 'DOCUMENT CONTEXT'.
        3. If the answer is not found in the document context, you MUST respond with the exact phrase: "The answer could not be found in the provided document(s)."
        4. Do not apologize, do not add any conversational fluff, do not explain that you are an AI, and do not introduce yourself.
        5. Be direct and concise. Get straight to the answer.
        """

        user_prompt = f"""
        **DOCUMENT CONTEXT:**
        ---
        {context}
        ---
        **QUESTION:** {question}
        """

        try:
            response = self.ollama_client.chat(
                model='llama3',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'temperature': 0.0} # Temperature 0 for more deterministic, factual answers
            )
            answer = response['message']['content'].strip()
            return answer
        except Exception as e:
            logging.error(f"Ollama Q&A failed: {e}")
            return "I'm sorry, but I encountered an error while trying to answer your question."
