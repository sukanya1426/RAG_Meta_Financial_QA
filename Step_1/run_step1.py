import os
from rag_pipeline import RAGPipeline
from impl import Datastore, Indexer, Retriever, ResponseGenerator

def run_step1():
    # Initialize pipeline
    pipeline = RAGPipeline(
        datastore=Datastore(),
        indexer=Indexer(),
        retriever=Retriever(datastore=Datastore()),
        response_generator=ResponseGenerator()
    )

    # Reset datastore
    print("🗑️ Resetting the database...")
    pipeline.reset()

    # Index Meta's Q1 2024 report
    meta_report_path = "data/source/meta_q1_2024.txt"  # Adjust if using PDF
    print(f"🔍 Adding document: {meta_report_path}")
    pipeline.add_documents([meta_report_path])

    # Test queries
    test_queries = [
        {"question": "What was Meta’s revenue in Q1 2024?", "answer": "$36.455 billion"},
        {"question": "What were the key financial highlights for Meta in Q1 2024?", 
         "answer": "Revenue: $36.455 billion (27% increase), Net income: $12.369 billion (117% increase), Operating margin: 38%, EPS: $4.71, DAP: 3.24 billion (7% increase)"}
    ]
    
    # Run queries and save outputs
    outputs = []
    for query in test_queries:
        response = pipeline.process_query(query["question"])
        outputs.append({
            "query": query["question"],
            "response": response,
            "expected_answer": query["answer"]
        })
        print(f"Query: {query['question']}\nResponse: {response}\nExpected: {query['answer']}\n")
    
    # Save outputs to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/step1_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)
    print("✅ Outputs saved to outputs/step1_outputs.json")

if __name__ == "__main__":
    run_step1()