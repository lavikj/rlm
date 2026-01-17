"""
Example: Using the document QA endpoint with RLM.

This demonstrates how to query a set of markdown documents using the
query_documents endpoint with an OpenRouter model.
"""

from dotenv import load_dotenv

from rlm.api import query_documents

load_dotenv()


def main():
    # Load a real (large) markdown doc as the long context.
    markdown_path = "example-docs/Synopsys/VCS/VCS User Guide 2019.06-SP1/markdown.md"
    with open(markdown_path, encoding="utf-8") as f:
        vcs_user_guide = f.read()

    documents = {"vcs_user_guide_markdown.md": vcs_user_guide}

    # The query to answer
    query = (
        "From the VCS User Guide markdown provided as context: "
        "(1) what is the guide's version and publication month/year, and "
        "(2) what is the name of the environment variable used for the "
        "synopsis_sim.setup file (setup file lookup)? Reply concisely."
    )

    print("=" * 60)
    print("Document QA Example with RLM")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print("Model: qwen/qwen3-235b-a22b-2507 (fixed, via Cerebras)")
    print(f"Documents: {list(documents.keys())}")
    print("\n" + "-" * 60)

    # Simple usage - just get the answer
    print("\n[Simple Usage - query_documents()]")
    try:
        answer = query_documents(
            documents=documents,
            query=query,
            verbose=False,  # Set to True to see RLM progress
        )
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "-" * 60)

    # # Detailed usage - get answer with statistics
    # print("\n[Detailed Usage - query_documents_detailed()]")
    # try:
    #     result = query_documents_detailed(
    #         documents=documents,
    #         query=query,
    #         model=model,
    #         verbose=False,
    #     )
    #     print(f"\nAnswer: {result['answer']}")
    #     print(f"Execution Time: {result['execution_time']:.2f}s")
    #     print(f"Usage: {result['usage']}")
    # except Exception as e:
    #     print(f"Error: {e}")


if __name__ == "__main__":
    main()
