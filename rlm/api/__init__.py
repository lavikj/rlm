"""
RLM API endpoints.

This module provides high-level endpoint functions for common RLM use cases.
"""

from rlm.api.document_qa import query_documents, query_documents_detailed

__all__ = ["query_documents", "query_documents_detailed"]
