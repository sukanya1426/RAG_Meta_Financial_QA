import glob
import json
import os
import sys
from pathlib import Path
from typing import List
from src.rag_pipeline import RAGPipeline
from create_parser import create_parser
from src.impl.datastore import Datastore
from src.impl.indexer import Indexer
from src.impl.retriever import Retriever
from src.impl.response_generator import ResponseGenerator
from src.impl.evaluator import Evaluator

DEFAULT_SOURCE_PATH = "data/source/Meta’s Q1 2024 Financial Report.pdf"
DEFAULT_EVAL_PATH = "sample_data/eval/sample_questions.json"

def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    try:
        datastore = Datastore()
        indexer = Indexer()
        retriever = Retriever(datastore=datastore)
        response_generator = ResponseGenerator()
        evaluator = Evaluator()
        return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)
    except Exception as e:
        print(f"❌ Failed to create pipeline: {str(e)}")
        raise

def get_files_in_directory(source_path: str) -> List[str]:
    """Get the Meta report file with debug information."""
    print(f"\n🔍 Checking file: {source_path}")
    
    try:
        path = Path(source_path).expanduser().resolve()
        if path.exists() and path.is_file() and path.suffix.lower() == '.pdf':
            print(f"📄 Found file: {path}")
            return [str(path)]
        
        alt_names = [
            "data/source/Meta's Q1 2024 Financial Report.pdf",
            "data/source/Meta Q1 2024 Financial Report.pdf",
            "data/source/Meta’s Q1 2024 Financial Report.pdf"
        ]
        for alt_path in alt_names:
            alt_path = Path(alt_path).expanduser().resolve()
            if alt_path.exists() and alt_path.is_file() and alt_path.suffix.lower() == '.pdf':
                print(f"📄 Found alternative file: {alt_path}")
                return [str(alt_path)]
        
        print(f"❌ File not found: {path}")
        print("❌ Please ensure 'data/source/Meta’s Q1 2024 Financial Report.pdf' exists.")
        return []
    
    except Exception as e:
        print(f"❌ Error accessing path: {str(e)}")
        return []

def execute_pipeline_commands(args, pipeline: RAGPipeline, document_paths: List[str], eval_path: str):
    """Execute pipeline commands based on the provided arguments."""
    if args.command in ["reset", "run"]:
        print("\n🗑️ Resetting the database...")
        pipeline.reset()

    if args.command in ["add", "run"]:
        if not document_paths:
            print("❌ No documents found to process! Ensure 'data/source/Meta’s Q1 2024 Financial Report.pdf' exists.")
            return
        print("\n📝 Adding documents:")
        for doc in document_paths:
            print(f"   Processing: {doc}")
        pipeline.add_documents(document_paths)
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

    if args.command in ["evaluate", "run"]:
        print(f"\n📊 Evaluating using questions from: {eval_path}")
        try:
            with open(eval_path, "r") as file:
                sample_questions = json.load(file)
            pipeline.evaluate(sample_questions)
        except FileNotFoundError:
            print(f"❌ Evaluation file not found: {eval_path}")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON format in evaluation file: {eval_path}")
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")

    if args.command == "query":
        if not args.prompt:
            print("❌ No query prompt provided!")
            return
        print(f"\n❓ Running query: {args.prompt}")
        try:
            response = pipeline.process_query(args.prompt)
            print(f"✨ Response: {response}")
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")

    elif args.command == "test_step2":
        print("\n🧪 Running Step 2 test queries (with hybrid search)...")
        print("\n🗑️ Resetting the database...")
        pipeline.reset()
        if not document_paths:
            print("❌ No documents found to process! Ensure 'data/source/Meta’s Q1 2024 Financial Report.pdf' exists.")
            return
        print("\n📝 Adding documents:")
        for doc in document_paths:
            print(f"   Processing: {doc}")
        pipeline.add_documents(document_paths)
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
                print(f"❌ Error processing test query: {str(e)}\n")
                outputs.append({
                    "query": query["question"],
                    "response": f"Error: {str(e)}",
                    "expected_answer": query["answer"]
                })
        
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/step2_outputs.json"
        try:
            with open(output_file, "w") as f:
                json.dump(outputs, f, indent=2)
            print(f"✅ Step 2 outputs saved to {output_file}")
        except Exception as e:
            print(f"❌ Error saving outputs: {str(e)}")

def main():
    """Main entry point of the script."""
    parser = create_parser()
    args = parser.parse_args()
    pipeline = create_pipeline()
    source_path = args.path if args.path else DEFAULT_SOURCE_PATH
    eval_path = args.eval_file if args.eval_file else DEFAULT_EVAL_PATH
    document_paths = get_files_in_directory(source_path)
    execute_pipeline_commands(args, pipeline, document_paths, eval_path)

if __name__ == "__main__":
    main()