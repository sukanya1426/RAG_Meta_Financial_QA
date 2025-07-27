import glob
import json
import os
from typing import List
from rag_pipeline import RAGPipeline
from Step_1.create_parser import create_parser
from impl import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator

DEFAULT_SOURCE_PATH = "data/source/"  # Directory for Meta's report
DEFAULT_EVAL_PATH = "sample_data/eval/sample_questions.json"

def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()  # Optional for Step 1
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)

def main():
    parser = create_parser()
    args = parser.parse_args()
    pipeline = create_pipeline()

    # Process source paths and eval path
    source_path = args.path if args.path else DEFAULT_SOURCE_PATH
    eval_path = args.eval_file if args.eval_file else DEFAULT_EVAL_PATH
    document_paths = get_files_in_directory(source_path)

    # Execute commands
    if args.command in ["reset", "run"]:
        print("🗑️ Resetting the database...")
        pipeline.reset()

    if args.command in ["add", "run"]:
        print(f"🔍 Adding documents: {', '.join(document_paths)}")
        pipeline.add_documents(document_paths)

    if args.command in ["evaluate", "run"]:
        print(f"📊 Evaluating using questions from: {eval_path}")
        with open(eval_path, "r") as file:
            sample_questions = json.load(file)
        pipeline.evaluate(sample_questions)

    if args.command == "query":
        print(f"✨ Response: {pipeline.process_query(args.prompt)}")
    elif args.command == "test_step1":
        # Run Step 1 test queries
        test_queries = [
            {"question": "What was Meta’s revenue in Q1 2024?", "answer": "$36.455 billion"},
            {"question": "What were the key financial highlights for Meta in Q1 2024?", 
             "answer": "Revenue: $36.455 billion (27% increase), Net income: $12.369 billion (117% increase), Operating margin: 38%, EPS: $4.71, DAP: 3.24 billion (7% increase)"}
        ]
        for query in test_queries:
            response = pipeline.process_query(query["question"])
            print(f"Query: {query['question']}\nResponse: {response}\nExpected: {query['answer']}\n")

def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    return glob.glob(os.path.join(source_path, "*"))

if __name__ == "__main__":
    main()