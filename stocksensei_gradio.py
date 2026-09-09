# stocksensei_gradio.py

import os
import gradio as gr
import pandas as pd
import yfinance as yf

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Ensure your OpenAI key is available
os.environ["OPENAI_API_KEY"] = ""  # <-- replace this with your key or set as env variable

# -------------------------
# Build the portfolio
# -------------------------
portfolio = pd.DataFrame({
    "Ticker": ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"],
    "Buy_Date": ["2024-01-15", "2024-02-01", "2024-01-25", "2024-02-10", "2024-01-05"],
    "Buy_Price": [185.0, 145.0, 220.0, 310.0, 500.0],
    "Quantity": [5, 3, 4, 2, 1]
})

# -------------------------
# Build retriever from Yahoo Finance
# -------------------------
def build_retriever(portfolio_df):
    stock_docs = []

    for idx, row in portfolio_df.iterrows():
        symbol = row['Ticker']
        buy_price = row['Buy_Price']
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1mo").reset_index()

        price_change = "N/A"
        perf_note = ""
        if len(hist) > 1:
            price_change_val = round((hist.iloc[-1]['Close'] - hist.iloc[0]['Close']) / hist.iloc[0]['Close'] * 100, 2)
            price_change = f"{price_change_val}%"
            if price_change_val < 0:
                perf_note = "The stock has declined this month, which could be dragging your portfolio."
            elif price_change_val < 2:
                perf_note = "The stock has remained mostly flat."
            else:
                perf_note = "The stock is performing well this month."

        summary = f"""
        Company: {symbol}
        Sector: {info.get('sector', 'N/A')}
        Industry: {info.get('industry', 'N/A')}
        Market Cap: {info.get('marketCap', 'N/A')}
        P/E Ratio: {info.get('trailingPE', 'N/A')}
        Dividend Yield: {info.get('dividendYield', 'N/A')}
        Weekly Performance: {price_change}
        Buy Price: {buy_price}
        Recent Close: {hist.iloc[-1]['Close'] if not hist.empty else 'N/A'}
        Performance Note: {perf_note}
        """
        stock_docs.append(Document(page_content=summary, metadata={"ticker": symbol}))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(stock_docs, embeddings)
    return vectorstore.as_retriever()

retriever = build_retriever(portfolio)

# -------------------------
# Setup LangChain RAG
# -------------------------
template = """
You are StockSensei, an intelligent assistant for financial queries.
Use only the provided context to answer questions. If unsure, say "I couldn't find that in the data."

Context:
{context}

Question: {question}
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------
# Define Gradio interface
# -------------------------
def answer_query(user_input):
    try:
        result = rag_chain.invoke({"query": user_input})
        return result["result"]
    except Exception as e:
        return f"🚫 Error: {str(e)}"

# Interface layout
iface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Ask StockSensei about your portfolio", placeholder="e.g., Why is my portfolio down this week?"),
    outputs=gr.Textbox(label="Answer"),
    title="📊 StockSensei – Financial RAG Chatbot",
    description="Ask questions like 'How is Nvidia performing?', 'Which stocks dropped?', etc."
)

if __name__ == "__main__":
    iface.launch()
