# StockSensei — Portfolio Assistant

A retrieval-augmented (RAG) chatbot that answers natural-language questions about your stock portfolio, grounded in live market data.

## How it works

1. **Live data pull** — for each ticker in your portfolio, fetches sector, industry, market cap, P/E ratio, dividend yield, and 1-month price history via `yfinance`.
2. **Document indexing** — turns each stock's summary into a document, embeds it with `sentence-transformers/all-MiniLM-L6-v2`, and indexes it in a FAISS vector store.
3. **Retrieval-augmented generation** — a LangChain `RetrievalQA` chain retrieves the relevant stock document(s) for a question and prompts an OpenAI LLM to answer using only that retrieved context (reduces hallucination on numbers).
4. **Answer evaluation** — every generated answer is scored against the underlying data: whether it names the right ticker, whether reported percentages/prices match the actual figures, and answer length — surfaced in an in-app metrics dashboard.
5. **UI** — a Gradio app showing your portfolio table, a question box with example prompts, and an expandable performance-metrics panel.

## Example questions

- "What's the best performing stock in my portfolio?"
- "How is NVDA doing compared to my purchase price?"
- "Which sector in my portfolio is performing best?"
- "Which of my stocks has the lowest P/E ratio?"

## Tech stack

- Python
- LangChain (`RetrievalQA`, `PromptTemplate`)
- FAISS (vector search)
- sentence-transformers (embeddings)
- yfinance (live market data)
- OpenAI API (LLM)
- Gradio (UI)

## Contents

```
StockSensei/
├─ StockSensei.ipynb        # notebook version: data fetch, indexing, RAG chain, evaluation, UI
├─ stocksensei_gradio.py    # standalone script version of the same app
└─ README.md
```

Two ways to run the same core idea:
- **`StockSensei.ipynb`** — the full notebook, including the answer-evaluation harness (scores generated answers against actual stock data) and a richer Gradio UI with a metrics dashboard and example questions.
- **`stocksensei_gradio.py`** — a lightweight standalone script version of the same RAG pipeline, for running outside a notebook (e.g. `python stocksensei_gradio.py`).

## How to run

### Notebook
1. Install dependencies:
   ```bash
   pip install langchain langchain-community langchain-openai faiss-cpu yfinance gradio sentence-transformers
   ```
2. Run the notebook. You'll be prompted to enter your OpenAI API key (entered securely via `getpass`, not stored in the notebook).
3. The app launches a Gradio interface (with a shareable public link via `demo.launch(share=True)`).
4. Ask questions about the sample portfolio (AAPL, GOOGL, TSLA, MSFT, NVDA) or edit the `portfolio` DataFrame to use your own holdings.

### Script
1. Install the same dependencies as above.
2. Set your OpenAI API key as an environment variable (never hardcode it in the file):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
3. Run:
   ```bash
   python stocksensei_gradio.py
   ```

## Notes

- The current portfolio is hardcoded as sample data — swap in real holdings by editing the `portfolio` DataFrame.
- Requires an OpenAI API key at runtime; none is stored in the notebook.
- The evaluation harness (`evaluate_response` / `calculate_metrics`) is a nice touch worth highlighting — it's not just a chatbot demo, it tracks whether the model's numeric claims actually match the retrieved data.
