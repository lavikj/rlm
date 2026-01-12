"""
Example: Using the document QA endpoint with RLM.

This demonstrates how to query a set of markdown documents using the
query_documents endpoint with an OpenRouter model.
"""

from dotenv import load_dotenv

from rlm.api import query_documents, query_documents_detailed

load_dotenv()


def main():
    # Example markdown documents
    documents = {
        "project_overview.md": """
# Project Alpha Overview

Project Alpha is an innovative AI-powered document analysis system developed in 2024.

## Key Features
- Semantic search across large document collections
- Multi-language support (English, Spanish, French, German)
- Real-time collaboration tools
- Export to PDF, DOCX, and HTML formats

## Technical Stack
- Backend: Python 3.12 with FastAPI
- Database: PostgreSQL 15 with pgvector
- Frontend: React 18 with TypeScript
- AI: Claude 3.5 Sonnet for document analysis
""",
        "team_info.md": """
# Team Information

## Core Team Members
- **Alice Chen** - Project Lead, joined January 2024
- **Bob Martinez** - Senior Backend Developer, joined February 2024
- **Carol Johnson** - Frontend Developer, joined March 2024
- **David Kim** - ML Engineer, joined January 2024

## Contact
- Email: team@projectalpha.example.com
- Slack: #project-alpha
""",
        "roadmap.md": """
# Development Roadmap

## Q1 2024 (Completed)
- [x] Initial prototype
- [x] Core API development
- [x] Basic UI implementation

## Q2 2024 (In Progress)
- [ ] Advanced search features
- [ ] User authentication
- [ ] API rate limiting

## Q3 2024 (Planned)
- [ ] Multi-tenant support
- [ ] Enterprise features
- [ ] Mobile app beta
""",
    }

    # The query to answer
    query = "What programming language and framework is used for the backend, and who is the senior backend developer?"

    # Specify the OpenRouter model
    model = "anthropic/claude-3.5-sonnet"

    print("=" * 60)
    print("Document QA Example with RLM")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"Model: {model}")
    print(f"Documents: {list(documents.keys())}")
    print("\n" + "-" * 60)

    # Simple usage - just get the answer
    print("\n[Simple Usage - query_documents()]")
    try:
        answer = query_documents(
            documents=documents,
            query=query,
            model=model,
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
