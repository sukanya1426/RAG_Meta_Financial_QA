import argparse


def create_parser():
    """Initialize and return the argument parser with all commands."""
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")

    # Define parent parsers for shared arguments
    # These will be used by both the main parser (by direct addition)
    # and relevant subparsers (via the `parents` attribute).

    path_arg_parent = argparse.ArgumentParser(add_help=False)
    path_arg_parent.add_argument(
        "-p",
        "--path",
        type=lambda x: x.strip('"\''),  # Strip quotes if present
        required=False,
        help="Path to a directory containing documents to index.",
    )

    eval_file_arg_parent = argparse.ArgumentParser(add_help=False)
    eval_file_arg_parent.add_argument(
        "-f",
        "--eval_file",
        type=str,
        required=False,
        help="Path to a .json file with question/expected_answer pairs.",
    )

    # Add global arguments to the main parser.
    # These definitions must match those in the parent parsers for consistent behavior.
    parser.add_argument(
        "-p",
        "--path",
        type=lambda x: x.strip('"\''),  # Strip quotes if present
        required=False,
        help="Path to a directory containing documents to index.",
    )

    parser.add_argument(
        "-f",
        "--eval_file",
        type=str,
        required=False,
        help="Path to a .json file with question/expected_answer pairs.",
    )

    # Then create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    subparsers.add_parser(
        "run",
        help="Run the full pipeline: reset, add, evaluate.",
        parents=[path_arg_parent, eval_file_arg_parent],
    )
    subparsers.add_parser("reset", help="Reset the database")
    subparsers.add_parser(
        "add", help="Add (index) documents to the database.", parents=[path_arg_parent]
    )
    subparsers.add_parser(
        "evaluate", help="Evaluate the model", parents=[eval_file_arg_parent]
    )

    # "Query" command
    query_parser = subparsers.add_parser("query", help="Query the documents")
    query_parser.add_argument("prompt", type=str, help="What to search for.")

    # "Test Step 1" command
    subparsers.add_parser(
        "test_step1",
        help="Run test queries for Step 1 evaluation"
    )

    # "Test Step 2" command
    subparsers.add_parser(
        "test_step2",
        help="Run test queries for Step 2 evaluation with hybrid search"
    )

    # "Test Step 3" command
    subparsers.add_parser(
        "test_step3",
        help="Run comprehensive Step 3 evaluation with advanced RAG features"
    )

    # "Demo Step 3" command
    subparsers.add_parser(
        "demo_step3",
        help="Run quick demo of Step 3 advanced features"
    )

    return parser
