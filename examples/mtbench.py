import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
import requests
from openai import OpenAI

# MT-bench dataset URL
MTBENCH_URL = ("https://raw.githubusercontent.com/lm-sys/FastChat/main/"
               "fastchat/llm_judge/data/mt_bench/question.jsonl")

# System prompt template for different models
PROMPT_TEMPLATES = {
    "gpt": ("A chat between a human and an AI assistant. "
            "The assistant gives helpful, detailed, accurate, "
            "and engaging responses.\n\nHuman: {instruction}\n\nAssistant:"),
    "llama": "<s>[INST] {instruction} [/INST]",
    "phi": "Instruct: {instruction}\nOutput:",
    # Add more model-specific templates as needed
}

# MTBench prompts and configurations
JUDGE_PROMPT = """You are a helpful AI assistant. You will be comparing \
two responses to a given prompt. Please compare them based on:
1. Helpfulness - How well does the response address the user's needs?
2. Relevance - How well does the response stay on topic?
3. Accuracy - Is the information provided correct and reliable?
4. Completeness - Does the response fully address all aspects of the prompt?

Rate each response on a scale of 1-10, where 10 is the best. Be critical \
and strict in your scoring.
Format your response as JSON with fields: "score_a", "score_b", "reasoning"
"""

AVAILABLE_MODELS = {
    'gpt2-large': 'gpt2-large',
    'Llama3.2-1B': 'meta-llama/Llama-3.2-1B-Instruct',
    'Llama3.2-3B': 'meta-llama/Llama-3.2-3B',
    'Phi3-mini-4k': 'microsoft/Phi-3-mini-4k-instruct',
    'Qwen2.5-3B': 'Qwen/Qwen2.5-3B-Instruct',
    'gemma-7B': 'google/gemma-7b-it'
}


class MTBenchEvaluator:
    def __init__(self, test_model_name="Llama3.2-1B"):
        if test_model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model {test_model_name} not found. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.test_model_name = test_model_name
        self.test_model_path = AVAILABLE_MODELS[test_model_name]
        # Set up OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        # Initialize test model only
        self.test_tokenizer = AutoTokenizer.from_pretrained(
            self.test_model_path
        )
        self.test_model = AutoModelForCausalLM.from_pretrained(
            self.test_model_path
        ).to(self.device)
        # Load MTBench dataset
        self.load_mtbench_data()

    def load_mtbench_data(self):
        """Load MTBench multi-turn conversation dataset"""
        # Download MT-bench questions if not exists
        cache_path = "mt_bench_questions.jsonl"
        if not os.path.exists(cache_path):
            response = requests.get(MTBENCH_URL)
            with open(cache_path, "w") as f:
                f.write(response.text)
        # Load conversations
        self.conversations = []
        with open(cache_path, "r") as f:
            for line in f:
                data = json.loads(line)
                turns = []
                for i, q in enumerate(data["turns"]):
                    turns.append({
                        "user": q,
                        "assistant": "..."
                    })
                self.conversations.append({
                    "id": data["question_id"],
                    "category": data["category"],
                    "turns": turns
                })

    def get_model_response(self, prompt: str) -> str:
        """Get response from test model"""
        # Get appropriate prompt template
        template = PROMPT_TEMPLATES.get(
            self.test_model_name.split("/")[-1].split("-")[0].lower(),
            PROMPT_TEMPLATES["gpt"]
        )
        formatted_prompt = template.format(instruction=prompt)
        inputs = self.test_tokenizer(
            formatted_prompt, return_tensors="pt"
        ).to(self.device)

        # Generate with error handling
        try:
            with torch.no_grad():
                outputs = self.test_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
            # Decode and return the generated response
            decode_outputs = self.test_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            if decode_outputs:
                return decode_outputs[0]
            else:
                print("No output generated.")
                return "No response generated"
            # return self.test_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error during generation"
        # outputs = self.test_model.generate(
        #     **inputs,
        #     max_length=2048,  # Increased for longer responses
        #     temperature=0.3,
        #     do_sample=True,
        #     top_p=0.95,
        #     repetition_penalty=1.2
        # )
        # return self.test_tokenizer.decode(outputs[0], skip_special_tokens=True)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    def get_judge_score(
            self, prompt: str, response_a: str, response_b: str
    ) -> Dict[str, Any]:
        """Get judgment scores using OpenAI's GPT model"""
        judge_input = (
            f"{JUDGE_PROMPT}\n\nPrompt: {prompt}\n\n"
            f"Response A: {response_a}\n\nResponse B: {response_b}\n\n"
            "Provide your evaluation as JSON:"
        )
        try:
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4.1-nano",  # or "gpt-3.5-turbo"
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI model evaluator."
                    },
                    {"role": "user", "content": judge_input}
                ],
                temperature=0.2,
                max_tokens=500
            )
            result = response.choices[0].message.content
            try:
                scores = json.loads(result)
                return scores
            except json.JSONDecodeError:
                return {
                    "score_a": 0,
                    "score_b": 0,
                    "reasoning": "Failed to parse GPT response"
                }
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return {
                "score_a": 0,
                "score_b": 0,
                "reasoning": f"API error: {str(e)}"
            }

    def evaluate_conversation(self, conversation: Dict) -> List[Dict]:
        """Evaluate a multi-turn conversation"""
        results = []
        context = ""
        for turn in conversation["turns"]:
            prompt = context + turn["user"]
            model_response = self.get_model_response(prompt)
            scores = self.get_judge_score(
                prompt, turn["assistant"], model_response
            )
            results.append({
                "prompt": turn["user"],
                "reference": turn["assistant"],
                "model_response": model_response,
                "scores": scores
            })
            
            context += f"User: {turn['user']}\nAssistant: {model_response}\n"
        return results

    def run_evaluation(self):
        """Run evaluation on all conversations"""
        all_results = []
        for conv in tqdm(self.conversations, desc="Evaluating conversations"):
            results = self.evaluate_conversation(conv)
            for r in results:
                r["category"] = conv["category"]
                r["question_id"] = conv["id"]
            all_results.extend(results)
            
        # Calculate statistics per category
        categories = set(r["category"] for r in all_results)
        stats = {}
        for category in categories:
            cat_results = [
                r for r in all_results if r["category"] == category
            ]
            scores_a = [r["scores"]["score_a"] for r in cat_results]
            scores_b = [r["scores"]["score_b"] for r in cat_results]
            stats[category] = {
                "mean_reference_score": np.mean(scores_a),
                "mean_model_score": np.mean(scores_b),
                "std_reference_score": np.std(scores_a),
                "std_model_score": np.std(scores_b)
            }
        
        # Overall stats
        scores_a = [r["scores"]["score_a"] for r in all_results]
        scores_b = [r["scores"]["score_b"] for r in all_results]
        stats["overall"] = {
            "mean_reference_score": np.mean(scores_a),
            "mean_model_score": np.mean(scores_b),
            "std_reference_score": np.std(scores_a),
            "std_model_score": np.std(scores_b)
        }
        
        return all_results, stats


def main():
    # Example usage with model selection
    model_name = "gpt2-large"  # Choose from AVAILABLE_MODELS
    evaluator = MTBenchEvaluator(test_model_name=model_name)
    
    results, stats = evaluator.run_evaluation()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("mtbench_results.csv", index=False)
    
    print("\nEvaluation Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    main()
