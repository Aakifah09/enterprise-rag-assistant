"""Ingestion error types surfaced to the user as readable messages."""


class IngestionError(Exception):
    """Base class for document ingestion problems."""


class UnsupportedFormatError(IngestionError):
    """The file extension or content type is not supported."""


class FileTooLargeError(IngestionError):
    """The upload exceeds the configured size limit."""


class ExtractionError(IngestionError):
    """The file could not be parsed."""


class EmptyDocumentError(IngestionError):
    """The file parsed but contains no extractable text."""


class DuplicateDocumentError(IngestionError):
    """An identical document (same content hash) is already indexed."""

    def __init__(self, message: str, existing_doc_id: str) -> None:
        super().__init__(message)
        self.existing_doc_id = existing_doc_id
