import os
import json
from src.rag_pipeline import RAGPipeline
from src.impl.datastore import Datastore
from src.impl.indexer import Indexer
from src.impl.retriever import Retriever
from src.impl.response_generator import ResponseGenerator

def run_step2():
    """Run the Step 2 RAG pipeline with enhanced structured data handling."""
    print("Initializing Step 2 pipeline with hybrid search capabilities...")
    
    # Create shared datastore instance
    datastore = Datastore()
    
    pipeline = RAGPipeline(
        datastore=datastore,
        indexer=Indexer(),
        retriever=Retriever(datastore=datastore),
        response_generator=ResponseGenerator()
    )

    print("🗑️ Resetting the database...")
    pipeline.reset()

    meta_report_path = "data/source/Meta’s Q1 2024 Financial Report.pdf"
    print(f"🔍 Adding document: {meta_report_path}")
    pipeline.add_documents([meta_report_path])

    print("\n🔍 Checking datastore content...")
    try:
        datastore_content = pipeline.datastore.table.to_pandas()
        print(f"Datastore contains {len(datastore_content)} items")
        if len(datastore_content) > 0:
            print("Sample content:", datastore_content['content'].iloc[0][:100], "...")
            # Check for structured data
            table_count = len(datastore_content[datastore_content['content_type'] == 'table'])
            text_count = len(datastore_content[datastore_content['content_type'] == 'text'])
            print(f"Content breakdown: {text_count} text chunks, {table_count} tables")
        else:
            print("⚠️ Warning: Datastore is empty. Check indexer logs for errors.")
    except Exception as e:
        print(f"❌ Error checking datastore: {str(e)}")

    # Step 2 test queries focusing on structured data and comparisons
    test_queries = [
        {
            "question": "What was Meta's revenue in Q1 2024?",
            "answer": "$36.455 billion"
        },
        {
            "question": "What was Meta's net income in Q1 2024 compared to Q1 2023?",
            "answer": "Q1 2024: $12.369 billion, Q1 2023: $5.709 billion (117% increase)"
        },
        {
            "question": "Summarize Meta's operating expenses in Q1 2024.",
            "answer": "Operating expenses breakdown including R&D, sales & marketing, and general & administrative costs"
        },
        {
            "question": "What were the key financial highlights for Meta in Q1 2024?",
            "answer": "Revenue: $36.455 billion (27% increase), Net income: $12.369 billion (117% increase), Operating margin: 38%, EPS: $4.71, DAP: 3.24 billion (7% increase)"
        }
    ]

    outputs = []
    for query in test_queries:
        print(f"\n❓ Test Query: {query['question']}")
        try:
            response = pipeline.process_query(query["question"])
            print(f"🤖 Response: {response}")
            print(f"✅ Expected: {query['answer']}\n")
            outputs.append({
                "query": query["question"],
                "response": response,
                "expected_answer": query["answer"]
            })
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            outputs.append({
                "query": query["question"],
                "response": f"Error: {str(e)}",
                "expected_answer": query["answer"]
            })

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/step2_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)
    print("✅ Step 2 outputs saved to outputs/step2_outputs.json")

if __name__ == "__main__":
    run_step2()