## FEATURE:

Evaluate the mt_bench score of common public models and using gpt-4.1-nano (and optional other gpt models) as judge

## EXAMPLES:

mtbench.py: score common model using gpt-4.1-nano the perplexity of gpt2-large, llama, phi3, qwen, gemma

REST/* : REST is a retrieval-based speculative decoding method designed to boost generation speed of LLMs.  Speed on HumanEval and mt_bench with standard autoregressive generation and REST.  => this task does NOT need to use REST, just reference the mt_bench evaluation on common models.

https://github.com/lm-sys/FastChat : use MT-bench, a set of challenging multi-turn open-ended questions to evaluate models.

https://github.com/mtbench101/mt-bench-101 : mt-bench including the dataset

## DOCUMENTATION:

[List out any documentation (web pages, sources for an MCP server like Crawl4AI RAG, etc.) that will need to be referenced during development]

## OTHER CONSIDERATIONS:


conda deactivate. Use uv to create virtual environment and install the required packages with cuda12.1 and keep cpu option for mac os

make sure to include pytest

I am using desktop GPU RTX 3060, there is limited memory.  Make sure to use flash attention whenever it applies.

I also use colab. After generate the code for desktop GPU, also generate the ipynb file to run on colab including the pip packages.