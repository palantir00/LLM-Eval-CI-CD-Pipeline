"""Generate a sample golden dataset (Q&A pairs about a generic digital bank) in JSONL format.

Run:  uv run python -m scripts.seed_golden_dataset

The domain is intentionally GENERIC (a typical banking / fintech app, no specific brand)
so that the project stays universal. The answers are ILLUSTRATIVE (a portfolio demo) and
describe common, standard banking practices — in a real system they should be replaced with
the actual policy of a given bank. The names in the context_sources field correspond to the
documents we will add to the knowledge base in Step 3.
"""

import json
import logging

from src.golden_dataset import Category, Difficulty, GoldenItem, load_golden_dataset
from src.paths import GOLDEN_DATASET_PATH

# Logging configuration: we use the logging module instead of print (project requirement).
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_golden_items() -> list[GoldenItem]:
    """Build the list of sample golden dataset entries.

    Returns:
        A list of 10 validated GoldenItem objects — 2 per each of the 5 topic categories.
    """
    return [
        # --- CARDS ---
        GoldenItem(
            id="bank-cards-01",
            question="What is the difference between a virtual card and a physical card?",
            expected_answer=(
                "A virtual card exists only in the app and is used mainly for online payments "
                "— it is available immediately after issuance. A physical card is a plastic "
                "card sent by post that can be used to pay in brick-and-mortar shops and to "
                "withdraw cash from ATMs."
            ),
            context_sources=["cards_policy.md"],
            category=Category.CARDS,
            difficulty=Difficulty.EASY,
            tags=["virtual card", "physical card", "payments"],
        ),
        GoldenItem(
            id="bank-cards-02",
            question="How do I block a lost or stolen card?",
            expected_answer=(
                "You can block the card instantly yourself in the app, in the card management "
                "section. The block takes effect immediately, and you can then order a new "
                "card if needed."
            ),
            context_sources=["cards_policy.md", "security_guide.md"],
            category=Category.CARDS,
            difficulty=Difficulty.MEDIUM,
            tags=["card blocking", "lost card", "security"],
        ),
        # --- TRANSFERS ---
        GoldenItem(
            id="bank-transfers-01",
            question="How long does a standard domestic transfer in the local currency take?",
            expected_answer=(
                "A standard domestic transfer is usually settled within a few hours on "
                "business days. Instant transfers (via the instant payment system) arrive "
                "within seconds, including outside the bank's working hours."
            ),
            context_sources=["transfers_guide.md"],
            category=Category.TRANSFERS,
            difficulty=Difficulty.EASY,
            tags=["domestic transfer", "settlement time", "instant payment"],
        ),
        GoldenItem(
            id="bank-transfers-02",
            question="What is the difference between a SEPA transfer and a SWIFT transfer?",
            expected_answer=(
                "A SEPA transfer covers euro payments within the SEPA area and is usually "
                "fast and cheap. A SWIFT transfer is used for international payments in various "
                "currencies outside the SEPA area; it can be slower and more expensive because "
                "intermediary bank fees may apply."
            ),
            context_sources=["transfers_guide.md"],
            category=Category.TRANSFERS,
            difficulty=Difficulty.MEDIUM,
            tags=["SEPA", "SWIFT", "international transfer"],
        ),
        # --- FEES ---
        GoldenItem(
            id="bank-fees-01",
            question="How much does it cost to maintain a basic account?",
            expected_answer=(
                "The basic account plan is usually free (no maintenance fee). Higher plans, "
                "offering additional benefits, are paid monthly according to the price list."
            ),
            context_sources=["fees_table.md"],
            category=Category.FEES,
            difficulty=Difficulty.MEDIUM,
            tags=["fees", "account maintenance", "pricing plans"],
        ),
        GoldenItem(
            id="bank-fees-02",
            question="How much is the fee for an international transfer?",
            expected_answer=(
                "The fee for an international transfer depends on the currency, amount, "
                "destination country and the chosen pricing plan — there is no single fixed "
                "amount. It is charged according to the current price list."
            ),
            context_sources=["fees_table.md"],
            category=Category.FEES,
            # HARD: the correct answer is "it depends" — a hallucination trap
            # (checks whether the model invents a specific fee amount).
            difficulty=Difficulty.HARD,
            tags=["commission", "international transfer", "hallucination-trap"],
        ),
        # --- ACCOUNT & VERIFICATION (KYC) ---
        GoldenItem(
            id="bank-account-01",
            question="What documents are required to open an account?",
            expected_answer=(
                "Opening an account usually requires a valid identity document (e.g. an ID "
                "card or passport) and a selfie for identity verification. The entire process "
                "takes place in the app."
            ),
            context_sources=["account_opening.md"],
            category=Category.ACCOUNT,
            difficulty=Difficulty.EASY,
            tags=["account opening", "documents", "onboarding"],
        ),
        GoldenItem(
            id="bank-account-02",
            question="What is KYC verification and why is it required?",
            expected_answer=(
                "KYC (Know Your Customer) is the process of confirming a customer's identity, "
                "required by anti-money-laundering (AML) regulations. The bank must verify the "
                "identity document and customer data before fully granting access to its services."
            ),
            context_sources=["account_opening.md", "security_guide.md"],
            category=Category.ACCOUNT,
            difficulty=Difficulty.MEDIUM,
            tags=["KYC", "identity verification", "AML"],
        ),
        # --- SECURITY ---
        GoldenItem(
            id="bank-security-01",
            question="What should I do if I notice an unauthorized transaction on my account?",
            expected_answer=(
                "You should block the card in the app as soon as possible and report the "
                "unauthorized transaction to the bank (e.g. via in-app chat or the helpline) "
                "to start the dispute procedure."
            ),
            context_sources=["security_guide.md"],
            category=Category.SECURITY,
            difficulty=Difficulty.EASY,
            tags=["unauthorized transaction", "dispute", "security"],
        ),
        GoldenItem(
            id="bank-security-02",
            question="How does two-factor authentication (2FA) work?",
            expected_answer=(
                "Two-factor authentication (2FA) requires, in addition to a password, a second "
                "element confirming identity — e.g. a one-time code from an SMS or a "
                "confirmation in the app. This makes access by unauthorized people harder even "
                "if the password is stolen."
            ),
            context_sources=["security_guide.md"],
            category=Category.SECURITY,
            difficulty=Difficulty.MEDIUM,
            tags=["2FA", "authentication", "security"],
        ),
    ]


def main() -> None:
    """Create the golden dataset, write it to a JSONL file and verify it by reloading."""
    items = build_golden_items()

    # Make sure the target directory exists (parents=True also creates parent directories).
    GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write in JSONL format: one entry = one JSON line.
    with GOLDEN_DATASET_PATH.open("w", encoding="utf-8") as file:
        for item in items:
            # model_dump_json() serializes the object to JSON.
            file.write(item.model_dump_json() + "\n")

    logger.info("Wrote %d entries to %s", len(items), GOLDEN_DATASET_PATH)

    # Round-trip verification: reload the file and check that it validates.
    reloaded = load_golden_dataset(GOLDEN_DATASET_PATH)
    logger.info("Verification: loaded and validated %d entries.", len(reloaded))

    # Short summary of the category distribution (useful to see topic coverage).
    counts: dict[str, int] = {}
    for item in reloaded:
        counts[item.category] = counts.get(item.category, 0) + 1
    logger.info("Category distribution: %s", json.dumps(counts))


if __name__ == "__main__":
    main()