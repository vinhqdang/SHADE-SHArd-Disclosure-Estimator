"""Multi-LLM QA pipeline (Experiment E8)."""
import os
import re
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Import SDKs with graceful fallbacks for testing
try:
    import openai
except ImportError:
    openai = None
    
try:
    import anthropic
except ImportError:
    anthropic = None
    
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class ClientRAG:
    """Simulates a single AI provider holding a single knowledge shard."""
    
    def __init__(self, provider: str, shard_docs: List[str], shard_embeddings: np.ndarray):
        """
        Args:
            provider: 'openai', 'anthropic', or 'gemini'
            shard_docs: List of documents in the provider's shard
            shard_embeddings: np.ndarray of embeddings for those documents
        """
        self.provider = provider
        self.shard_docs = shard_docs
        self.shard_embeddings = shard_embeddings
        
        # Determine which model to use
        if provider == "openai":
            self.model = "gpt-4o-mini"
            self.client = openai.Client() if openai and "OPENAI_API_KEY" in os.environ else None
        elif provider == "anthropic":
            self.model = "claude-3-haiku-20240307"
            self.client = anthropic.Anthropic() if anthropic and "ANTHROPIC_API_KEY" in os.environ else None
        elif provider == "gemini":
            self.model = "gemini-1.5-flash"
            if genai and "GEMINI_API_KEY" in os.environ:
                genai.configure(api_key=os.environ["GEMINI_API_KEY"])
                self.client = "gemini"
            else:
                self.client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """Dense retrieval of top-k documents from the shard."""
        if len(self.shard_docs) == 0:
            return []
            
        k = min(top_k, len(self.shard_docs))
        sims = cosine_similarity([query_embedding], self.shard_embeddings)[0]
        top_indices = np.argsort(sims)[-k:][::-1]
        
        return [self.shard_docs[i] for i in top_indices]
        
    def answer(self, query: str, query_embedding: np.ndarray) -> str:
        """Retrieve relevant context and generate an answer using the provider's API."""
        retrieved_docs = self.retrieve(query_embedding)
        context = "\\n---\\n".join(retrieved_docs)
        
        prompt = f"""You are an AI assistant. Answer the user's question based strictly on the provided context. If the context does not contain the answer, say "I don't know based on my knowledge base."
        
Context:
{context}

Question: {query}
Answer:"""

        # Mock response if API key/client is missing (for local testing without all 3 keys)
        if not self.client:
            return f"[Mock {self.provider} Answer] Based on my shard: {context[:50]}..."
            
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                return response.content[0].text.strip()
                
            elif self.provider == "gemini":
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                return response.text.strip()
                
        except Exception as e:
            return f"[API Error from {self.provider}: {str(e)}]"


class MultiLLMFusion:
    """Coordinates multiple ClientRAG instances and fuses their answers."""
    
    def __init__(self, clients: List[ClientRAG]):
        self.clients = clients
        self.fusion_client = openai.Client() if openai and "OPENAI_API_KEY" in os.environ else None
        
    def fuse_answers(self, query: str, query_embedding: np.ndarray) -> Dict[str, str]:
        """
        Query all clients and fuse their responses.
        
        Returns:
            Dict containing individual answers and the final fused answer.
        """
        results = {}
        answers_text = ""
        
        for i, client in enumerate(self.clients):
            ans = client.answer(query, query_embedding)
            results[client.provider] = ans
            answers_text += f"Provider {client.provider}:\n{ans}\n\n"
            
        fusion_prompt = f"""You are an expert consensus system. You have asked multiple independent AI providers the following question:
Question: {query}

Here are their answers (based on their distinct, private knowledge shards):
{answers_text}

Synthesize these answers into a single, accurate final answer. Give more weight to answers that cite specific facts or details. If all say they don't know, state that.
Return ONLY the final synthesized answer."""

        if not self.fusion_client:
            results['fused'] = "[Mock Fused Answer] Consensus: " + answers_text[:100].replace("\\n", " ") + "..."
            return results
            
        try:
            response = self.fusion_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fusion_prompt}],
                temperature=0.0,
                max_tokens=200
            )
            results['fused'] = response.choices[0].message.content.strip()
        except Exception as e:
            results['fused'] = f"[Fusion API Error: {str(e)}]"
            
        return results

def compute_exact_match(prediction: str, gold: str) -> float:
    """Compute exact match between prediction and gold answer."""
    def normalize_answer(s: str) -> str:
        s = s.lower()
        s = re.sub(r'\\b(a|an|the)\\b', ' ', s)
        s = re.sub(r'[^a-z0-9]', ' ', s)
        return ' '.join(s.split())
        
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0

def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold answer."""
    def normalize_tokens(s: str) -> List[str]:
        s = s.lower()
        s = re.sub(r'[^a-z0-9]', ' ', s)
        return s.split()
        
    pred_tokens = normalize_tokens(prediction)
    gold_tokens = normalize_tokens(gold)
    
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
        
    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0
        
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)
