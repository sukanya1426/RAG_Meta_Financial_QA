#!/usr/bin/env python3
"""
Reduced Step 3 Evaluation Script - Minimal API Usage
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import RAGPipeline
from quota_config import quota_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_minimal_evaluation():
    """Run a minimal evaluation with quota constraints"""
    logger.info("🚀 Starting Step 3 Minimal Evaluation...")
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline()
    
    # Minimal test queries (only 3 to save quota)
    test_queries = [
        "What was Meta's total revenue in Q1 2024?",
        "How many daily active people did Meta have in Q1 2024?", 
        "What was Meta's net income in Q1 2024?"
    ]
    
    results = {
        "evaluation_type": "minimal",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(test_queries),
        "results": [],
        "summary": {}
    }
    
    successful_queries = 0
    total_time = 0
    
    for i, query in enumerate(test_queries):
        logger.info(f"📝 Processing query {i+1}/{len(test_queries)}: {query}")
        
        try:
            start_time = time.time()
            
            # Test basic retrieval (no optimization to save API calls)
            logger.info("  🔍 Testing basic retrieval...")
            retrieved_docs = pipeline.retriever.search(query, top_k=5)
            
            # Test response generation with minimal context
            logger.info("  🤖 Generating response...")
            response = pipeline.response_generator.generate_response(
                query=query,
                context={
                    'text_context': [doc.content for doc in retrieved_docs[:3]],  # Limit context
                    'financial_data': [],
                    'structured_data': []
                }
            )
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Basic evaluation
            is_successful = (
                response and 
                len(response) > 50 and 
                "cannot find" not in response.lower() and
                "error" not in response.lower()
            )
            
            if is_successful:
                successful_queries += 1
            
            result = {
                "query": query,
                "response": response,
                "successful": is_successful,
                "response_time": elapsed_time,
                "retrieved_docs_count": len(retrieved_docs)
            }
            
            results["results"].append(result)
            logger.info(f"  ✅ Query completed in {elapsed_time:.2f}s - Success: {is_successful}")
            
            # Add delay to respect rate limits
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"  ❌ Error processing query: {e}")
            result = {
                "query": query,
                "response": f"Error: {str(e)}",
                "successful": False,
                "response_time": 0,
                "retrieved_docs_count": 0
            }
            results["results"].append(result)
    
    # Calculate summary metrics
    success_rate = successful_queries / len(test_queries) if test_queries else 0
    avg_response_time = total_time / len(test_queries) if test_queries else 0
    
    results["summary"] = {
        "success_rate": success_rate,
        "successful_queries": successful_queries,
        "total_queries": len(test_queries),
        "average_response_time": avg_response_time,
        "total_time": total_time
    }
    
    # Save results
    output_file = Path("outputs/step3_minimal_evaluation.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("                STEP 3: MINIMAL RAG EVALUATION SUMMARY")
    print("="*80)
    print(f"📊 EVALUATION OVERVIEW:")
    print(f"   Total Queries Evaluated: {len(test_queries)}")
    print(f"   Successful Queries: {successful_queries}")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Average Response Time: {avg_response_time:.2f}s")
    print(f"   Total Evaluation Time: {total_time:.2f}s")
    print()
    
    print("📋 QUERY RESULTS:")
    for i, result in enumerate(results["results"], 1):
        status = "✅" if result["successful"] else "❌"
        print(f"   {i}. {status} {result['query'][:60]}...")
        if result["successful"]:
            print(f"      Response: {result['response'][:100]}...")
        print(f"      Time: {result['response_time']:.2f}s")
        print()
    
    print("💡 RECOMMENDATIONS:")
    if success_rate >= 0.8:
        print("   ✅ System is working well! Ready for full evaluation when quota allows.")
    elif success_rate >= 0.5:
        print("   ⚠️  System partially working. Check API connectivity and data.")
    else:
        print("   ❌ System needs attention. Check configuration and logs.")
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = run_minimal_evaluation()
        sys.exit(0 if results["summary"]["success_rate"] > 0.5 else 1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
