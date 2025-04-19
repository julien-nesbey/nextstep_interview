import logging
import time
import os
import re
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, List, TypeVar, Tuple

from gql import Client
from gql.transport.requests import RequestsHTTPTransport
from sentence_transformers import SentenceTransformer, util
import json
from graphQL import (
    START_INTERVIEW,
    SAVE_INTERVIEW_DATA,
    END_INTERVIEW,
    GET_INTERVIEW_DATA,
    SAVE_INTERVIEW_ANALYSIS,
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")

# GraphQL client configuration
MAX_RETRIES = 3
BASE_DELAY = 1  # seconds
MAX_DELAY = 10  # seconds

# Initialize GraphQL client with retry configuration
transport = RequestsHTTPTransport(
    url="http://51.20.143.187/graphql",
    headers={"Content-Type": "application/json"},
    use_json=True,
    retries=3,
)

client = Client(
    transport=transport, fetch_schema_from_transport=True, execute_timeout=30
)


def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to implement retry logic with exponential backoff"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {MAX_RETRIES} attempts failed for {func.__name__}"
                    )
        raise last_exception

    return wrapper


@with_retry
def startInterview(user_id: str, role: str) -> str:
    """Start a new interview session.

    Args:
        user_id: The ID of the user starting the interview
        role: The role being interviewed for

    Returns:
        str: The ID of the created interview

    Raises:
        Exception: If the interview creation fails
    """
    try:
        result = client.execute(
            START_INTERVIEW, variable_values={"userId": user_id, "role": role}
        )
        interview_id = result["startInterview"]["id"]
        logger.info(f"Started interview {interview_id} for user {user_id}, role {role}")
        return interview_id
    except Exception as e:
        logger.error(f"Failed to start interview for user {user_id}: {str(e)}")
        raise


@with_retry
def saveInterviewData(
    interviewId: str, questions: List[str], answers: List[str]
) -> bool:
    """Save interview questions and answers.

    Args:
        interviewId: The ID of the interview
        questions: List of interview questions
        answers: List of user answers

    Returns:
        bool: True if save was successful
    """
    try:
        logger.info(f"Attempting to save interview data for {interviewId}")
        logger.info(f"Questions count: {len(questions)}, Answers count: {len(answers)}")

        if not questions or not answers:
            logger.error("Empty questions or answers list, skipping save")
            return False

        if len(questions) != len(answers):
            logger.error(
                f"Questions and answers count mismatch: {len(questions)} vs {len(answers)}"
            )

        # Print first question and answer for debugging
        if questions and answers:
            logger.info(f"First question sample: {questions[0][:50]}...")
            logger.info(f"First answer sample: {answers[0][:50]}...")

        # Try to validate the interview ID first
        try:
            interview_check = client.execute(
                GET_INTERVIEW_DATA, variable_values={"interviewId": interviewId}
            )
            if not interview_check or not interview_check.get("getInterviewData"):
                logger.error(f"Interview ID {interviewId} doesn't exist in database")
        except Exception as e:
            logger.error(f"Interview ID validation error: {str(e)}")

        # Prepare data payload
        data_payload = {
            "data": {
                "interviewId": interviewId,
                "question": questions,
                "answer": answers,
            }
        }

        logger.info("Sending data payload to GraphQL API")

        # Execute the GraphQL mutation
        result = client.execute(SAVE_INTERVIEW_DATA, variable_values=data_payload)

        success = bool(result.get("saveInterviewData", False))
        if success:
            logger.info(f"Successfully saved data for interview {interviewId}")
        else:
            logger.error(f"Save operation returned false for {interviewId}: {result}")

        return success
    except Exception as e:
        logger.error(f"Failed to save data for interview {interviewId}: {str(e)}")
        logger.exception("Detailed error trace:")
        raise


@with_retry
def getInterviewData(interviewId: str) -> Dict[str, Any]:
    """Retrieve interview data.

    Args:
        interviewId: The ID of the interview

    Returns:
        Dict containing interview data
    """
    try:
        result = client.execute(
            GET_INTERVIEW_DATA, variable_values={"interviewId": interviewId}
        )
        data = result.get("getInterviewData", {})
        logger.debug(f"Retrieved data for interview {interviewId}")
        return data
    except Exception as e:
        logger.error(f"Failed to retrieve data for interview {interviewId}: {str(e)}")
        raise


@with_retry
def saveInterviewAnalysis(interviewId: str, analysis: str) -> Dict[str, Any]:
    """Save interview analysis results.

    Args:
        interviewId: The ID of the interview
        analysis: Analysis results to save

    Returns:
        Dict containing save confirmation
    """
    try:
        result = client.execute(
            SAVE_INTERVIEW_ANALYSIS,
            variable_values={"interviewId": interviewId, "analysis": analysis},
        )
        response = result.get("saveInterviewAnalysis", {})
        logger.info(f"Saved analysis for interview {interviewId}")
        return response
    except Exception as e:
        logger.error(f"Failed to save analysis for interview {interviewId}: {str(e)}")
        raise


@with_retry
def endInterview(interviewId: str) -> bool:
    """End an interview session.

    Args:
        interviewId: The ID of the interview to end

    Returns:
        bool: True if interview was ended successfully
    """
    try:
        result = client.execute(
            END_INTERVIEW, variable_values={"interviewId": interviewId}
        )
        success = bool(result.get("endInterview", False))
        if success:
            logger.info(f"Successfully ended interview {interviewId}")
        return success
    except Exception as e:
        logger.error(f"Failed to end interview {interviewId}: {str(e)}")
        raise


# Fast regex patterns for common rephrase requests
REPHRASE_PATTERNS = [
    r"(?i)can you (please |kindly )?(repeat|rephrase|say (that|it) again)",
    r"(?i)i (didn't|did not) (understand|hear|get) (that|the question)",
    r"(?i)what do you mean",
    r"(?i)sorry.{0,10}(repeat|rephrase|unclear)",
    r"(?i)please (repeat|explain|clarify)",
    r"(?i)could you (clarify|explain)",
    r"(?i)(don't|do not) understand",
    r"(?i)better explanation",
    r"(?i)sorry.{0,5}(what|huh|pardon)",
    r"(?i)can you (make it|be) (clearer|simpler)",
]


# Cache the loaded model and triggers
@lru_cache(maxsize=1)
def _load_rephrase_resources() -> Tuple[Any, List[str], Any]:
    """Load and cache the sentence transformer model and triggers.

    Returns:
        Tuple containing (model, intent_list, intent_embeddings)
    """
    start_time = time.time()
    try:
        # Use smaller, faster model that's still effective
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Try multiple paths to find the trigger file
        potential_paths = [
            os.path.join(os.getcwd(), "common", "rephrase_triggers.json"),
            os.path.join(
                os.getcwd(), "src", "interview_app", "common", "rephrase_triggers.json"
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "common",
                "rephrase_triggers.json",
            ),
        ]

        data = None
        for path in potential_paths:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                logger.info(f"Loaded rephrase triggers from {path}")
                break

        if data is None:
            # Fallback default triggers if file not found
            logger.warning("Rephrase triggers file not found, using defaults")
            intents = [
                "Could you repeat that?",
                "I didn't understand the question",
                "Can you rephrase that?",
                "Please explain again",
                "What do you mean by that?",
                "I didn't get the question",
                "Can you be more clear?",
                "Sorry, I didn't hear you",
            ]
        else:
            intents = data["rephrase_triggers"]

        # Pre-compute embeddings
        intent_embs = model.encode(intents, convert_to_tensor=True)

        load_time = time.time() - start_time
        logger.info(f"Loaded rephrase detection resources in {load_time:.2f}s")

        return model, intents, intent_embs
    except Exception as e:
        logger.error(f"Error loading rephrase resources: {str(e)}")
        # Return minimal working resources in case of error
        dummy_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        dummy_intents = ["Can you repeat that?"]
        dummy_embeds = dummy_model.encode(dummy_intents, convert_to_tensor=True)
        return dummy_model, dummy_intents, dummy_embeds


def detect_rephrase(text: str) -> bool:
    """Detect if a user is asking for a question to be rephrased.

    Uses a combination of regex pattern matching (fast) and semantic similarity (accurate)
    for better performance and accuracy.

    Args:
        text: The user's message text

    Returns:
        bool: True if the text is likely requesting a rephrasing
    """
    if not text:
        return False

    start_time = time.time()

    try:
        # Fast pattern matching first
        for pattern in REPHRASE_PATTERNS:
            if re.search(pattern, text):
                logger.info(f"Rephrase detected via pattern matching: '{text[:50]}...'")
                return True

        # For short texts, semantic matching may be more helpful
        if len(text) < 100:
            # Load cached resources
            model, intents, intent_embs = _load_rephrase_resources()

            # Encode user text
            user_emb = model.encode(text, convert_to_tensor=True)

            # Calculate similarities
            similarities = util.cos_sim(user_emb, intent_embs)
            max_sim = similarities.max().item()

            # Return result based on threshold
            is_rephrase = max_sim >= 0.65  # Higher threshold for better precision

            proc_time = time.time() - start_time
            logger.debug(
                f"Rephrase detection took {proc_time:.3f}s, max similarity: {max_sim:.3f}"
            )

            if is_rephrase:
                logger.info(
                    f"Rephrase detected via semantic matching: '{text[:50]}...'"
                )

            return is_rephrase

        return False
    except Exception as e:
        logger.error(f"Error in rephrase detection: {str(e)}")
        return False  # Default to not rephrasing on error
