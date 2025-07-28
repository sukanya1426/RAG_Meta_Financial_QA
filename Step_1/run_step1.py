import os
import json
from src.rag_pipeline import RAGPipeline
from src.impl.datastore import Datastore
from src.impl.indexer import Indexer
from src.impl.retriever import Retriever
from src.impl.response_generator import ResponseGenerator

def run_step1():
    """Run the Step 1 RAG pipeline for Meta's Q1 2024 financial report."""
    print("Initializing pipeline...")
    
    # Create shared datastore instance
    datastore = Datastore()
    
    pipeline = RAGPipeline(
        datastore=datastore,
        indexer=Indexer(),
        retriever=Retriever(datastore=datastore),  # Use the same datastore
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
        else:
            print("⚠️ Warning: Datastore is empty. Check indexer logs for errors.")
    except Exception as e:
        print(f"❌ Error checking datastore: {str(e)}")

    test_queries = [
        {
            "question": "What was Meta's revenue in Q1 2024?",
            "answer": "$36.455 billion"
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
    with open("outputs/step1_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)
    print("✅ Outputs saved to outputs/step1_outputs.json")

if __name__ == "__main__":
    run_step1()