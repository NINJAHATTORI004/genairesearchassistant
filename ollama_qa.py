import ollama
from typing import Dict, Optional
import re

class OllamaQA:
    def __init__(self, model_name: str = "llama3:instruct"):
        """
        Initialize the Ollama QA model
        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3:instruct', 'mistral')
        """
        self.model_name = model_name
        self.system_prompt = """You are a helpful AI assistant that provides accurate, detailed answers based on the given context. 
        Follow these guidelines:
        1. Answer the question using only the information from the provided context
        2. Be precise and include relevant details
        3. If the context doesn't contain enough information, say so
        4. Format your response in clear, readable markdown
        """
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Clean and format the model's response"""
        # Remove any system-like prompts that might be in the response
        response = re.sub(r'^.*?(?=Answer:|$)', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'^Answer:', '', response).strip()
        return response
    
    def ask_question(self, context: str, question: str) -> Dict:
        """
        Ask a question about the given context using Ollama
        
        Args:
            context: The document or text to answer questions about
            question: The question to answer
            
        Returns:
            Dict containing the answer and metadata
        """
        try:
            
            prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a detailed and accurate answer. If the context doesn't contain enough information, say so.
            Answer: """
            
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_ctx': 4096
                }
            )
            
         
            answer = response['response'].strip()
            
           
            if not answer:
                answer = "I couldn't generate a response. The model returned an empty answer."
            
            return {
                'answer': answer,
                'context': context[:1000] + ('...' if len(context) > 1000 else ''),
                'highlight': answer[:200],
                'confidence': 90.0 if answer else 0,
                'is_comprehensive': True,
                'model': self.model_name
            }
            
        except Exception as e:
            return {
                'answer': f"Error getting response from Ollama: {str(e)}\n\nMake sure Ollama is running and the model is downloaded.",
                'context': "",
                'highlight': "",
                'confidence': 0,
                'is_comprehensive': False,
                'model': self.model_name
            }


if __name__ == "__main__":
   
    qa = OllamaQA(model_name="mistral")
    
    context = """
    Transformers are a type of neural network architecture that has become fundamental in natural language processing. 
    They were introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017.
    """
    
    question = "What is a transformer?"
    
   
    result = qa.ask_question(context, question)
    print(f"Answer: {result['answer']}")
    print(f"Model: {result.get('model', 'N/A')}")

