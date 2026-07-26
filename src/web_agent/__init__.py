"""Verified Protocol III web agent."""

from .contracts import ActionReceipt, CoverageCertificate, EvidenceRef, TaskContract
from .verifier import CompletionVerifier, VerificationResult

__all__ = [
    "ActionReceipt",
    "CompletionVerifier",
    "CoverageCertificate",
    "EvidenceRef",
    "TaskContract",
    "VerificationResult",
]

