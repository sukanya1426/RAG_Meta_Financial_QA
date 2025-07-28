#!/usr/bin/env python3
"""
Debug script to examine what context is being passed to the response generator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('.'))))
from src.impl.datastore import Datastore
from src.impl.indexer import Indexer
from src.impl.retriever import AdvancedRetriever
from src.impl.response_generator import ResponseGenerator
from src.impl.evaluator import Evaluator
from src.rag_pipeline import RAGPipeline
import logging
import json

logging.basicConfig(level=logging.WARNING)  # Reduce noise

def main():
    # Initialize components
    datastore = Datastore()
    indexer = Indexer()
    retriever = AdvancedRetriever(datastore=datastore, enable_optimization=True, enable_reranking=True)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()

    # Test query
    query = 'What was Meta total revenue in Q1 2024?'
    print(f"🔍 Testing query: {query}")
    print("=" * 80)

    # Get search results directly from retriever
    print("\n📊 Step 1: Getting search results...")
    search_results = retriever.hybrid_search(query)
    
    print(f"\n📋 Search Results Structure:")
    for key, value in search_results.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
            if value and len(value) > 0:
                print(f"    First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else 'Not a dict'}")
        else:
            print(f"  {key}: {type(value)}")

    # Show text context
    text_context = search_results.get('text_context', [])
    print(f"\n📝 Text Context ({len(text_context)} items):")
    for i, context in enumerate(text_context[:3]):  # Show first 3
        print(f"  [{i+1}] {context[:200]}...")

    # Show financial data
    financial_data = search_results.get('financial_data', [])
    print(f"\n💰 Financial Data ({len(financial_data)} items):")
    for i, data in enumerate(financial_data[:3]):  # Show first 3
        if isinstance(data, dict):
            content = data.get('content', 'No content')
            print(f"  [{i+1}] Type: {data.get('type', 'unknown')}")
            print(f"      Content: {content[:200]}...")
        else:
            print(f"  [{i+1}] {str(data)[:200]}...")

    # Show structured data
    structured_data = search_results.get('structured_data', [])
    print(f"\n🏗️ Structured Data ({len(structured_data)} items):")
    for i, data in enumerate(structured_data[:2]):  # Show first 2
        print(f"  [{i+1}] {str(data)[:200]}...")

    # Now let's see what the response generator would receive
    print(f"\n🤖 Step 2: Testing response generation...")
    
    # Test with a simple direct question first
    print(f"\n💭 Direct context test:")
    if text_context:
        # Use just the text context directly
        simple_response = response_generator.generate_response(query, text_context[:3])
        print(f"Simple response: {simple_response}")
    
    # Test with hybrid response
    print(f"\n🔄 Hybrid response test:")
    hybrid_response = response_generator.generate_hybrid_response(query, search_results)
    print(f"Hybrid response: {hybrid_response}")

    print("\n" + "=" * 80)
    print("🔍 Analysis Complete!")

if __name__ == "__main__":
    main()
