"""Schema and validation for the golden dataset (question-answer pairs).

The golden dataset is our "answer key": a set of questions with reference answers
that we compare model outputs against during evaluation. Here we define what a single
entry looks like (a Pydantic model) and how to safely load the whole set from a file.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Category(StrEnum):
    """Topic category of a question (a closed list of allowed values).

    Domain: a generic digital bank / fintech app (no specific brand).
    """

    CARDS = "cards"          # payment cards: virtual, physical, limits, blocking
    TRANSFERS = "transfers"  # transfers: domestic, SEPA, SWIFT, settlement time
    FEES = "fees"            # fees and commissions
    ACCOUNT = "account"      # account opening and identity verification (KYC)
    SECURITY = "security"    # security: 2FA, disputes, unauthorized transactions


class Difficulty(StrEnum):
    """Question difficulty level — lets us analyze results broken down by difficulty."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoldenItem(BaseModel):
    """A single golden dataset entry: one question with its reference answer.

    Pydantic automatically checks types and constraints when the object is created —
    if the data is invalid, we get a clear error instead of a silent failure later on.

    Attributes:
        id: Unique identifier of the entry (e.g. "bank-cards-01").
        question: The question asked to the model.
        expected_answer: The reference, correct answer (the baseline for metrics).
        context_sources: Names of knowledge base documents that contain the answer
            (used when scoring faithfulness — whether the model stuck to the sources).
        category: Topic category (from the Category list).
        difficulty: Difficulty level (from the Difficulty list).
        tags: Free-form helper labels for filtering/analysis.
    """

    # extra="forbid" => unknown fields (e.g. a typo "questionn") raise a validation error
    # instead of being silently ignored. Catches data mistakes as early as possible.
    model_config = ConfigDict(extra="forbid")

    # min_length=1 => these fields cannot be an empty string.
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    # min_length=1 => every question must reference at least one source.
    context_sources: list[str] = Field(min_length=1)
    category: Category
    difficulty: Difficulty
    # default_factory=list => defaults to an empty list (tags are optional).
    tags: list[str] = Field(default_factory=list)


def load_golden_dataset(path: Path) -> list[GoldenItem]:
    """Load and validate the golden dataset from a JSONL file.

    Reads the file line by line (each line is one entry in JSON format) and validates
    each entry separately. This way, in case of an error we know EXACTLY which line is
    the problem — the main advantage of JSONL over one large JSON file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        A list of validated GoldenItem entries.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When a line is invalid or an id is duplicated.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    items: list[GoldenItem] = []
    seen_ids: set[str] = set()  # used to detect duplicate identifiers

    with path.open(encoding="utf-8") as file:
        # enumerate(..., start=1) numbers lines from 1 (as a human sees them in an editor).
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue  # skip empty lines (e.g. a trailing newline at the end of file)

            try:
                # model_validate_json parses JSON and validates the schema in one step.
                item = GoldenItem.model_validate_json(line)
            except ValidationError as error:
                # Add the line number to the message — easier to locate the bad data.
                raise ValueError(
                    f"Validation error on line {line_number} of {path}:\n{error}"
                ) from error

            if item.id in seen_ids:
                raise ValueError(
                    f"Duplicate id '{item.id}' on line {line_number} of {path}"
                )
            seen_ids.add(item.id)
            items.append(item)

    return items