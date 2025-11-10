from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

import firebase_admin
import fitz
import mammoth
from better_profanity import profanity
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials
from google import genai
from google.cloud import firestore
from livekit import api as lk_api

load_dotenv()

logger = logging.getLogger(__name__)

TRUSTED_ORIGINS = ["https://prep.kprime.dev", "http://localhost:3000", "http://127.0.0.1:3000"]

ALLOWED_GRADES = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
}
ALLOWED_AVATARS = {"avatar", "avatar1", "avatar2", "avatar3", "avatar4"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_TEXT_LENGTH_CHARS = 20_000
STUDY_INPUT_MODES = {"file", "text"}
PDF_MIME_TYPES = {"application/pdf"}
DOCX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
}
SESSION_COLLECTION = "session_configs"

BANNED_TERMS_PATH = Path(__file__).with_name("banned_terms.txt")

def _load_strict_terms() -> set[str]:
    terms: set[str] = set()
    if not BANNED_TERMS_PATH.exists():
        return terms

    for raw_line in BANNED_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lower()
        if not line or line.startswith("#"):
            continue
        terms.add(line)
    return terms


def _load_allowed_origins() -> list[str]:
    configured = os.getenv("SESSION_API_ALLOWED_ORIGINS")
    if not configured:
        return TRUSTED_ORIGINS
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


STRICT_DENY_TERMS = _load_strict_terms()
STRICT_DENY_PATTERNS = [re.compile(rf"\b{re.escape(term)}\b") for term in STRICT_DENY_TERMS]

profanity.load_censor_words()
if STRICT_DENY_TERMS:
    profanity.add_censor_words(list(STRICT_DENY_TERMS))

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

_FIREBASE_PROJECT_ENV_KEYS = (
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "PROJECT_ID",
)


def _resolve_firebase_project_id() -> str:
    for key in _FIREBASE_PROJECT_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value
    keys = ", ".join(_FIREBASE_PROJECT_ENV_KEYS)
    raise RuntimeError(
        f"Missing Firebase project configuration. Set one of the following environment variables: {keys}."
    )

# def _initialize_firebase_app() -> firebase_admin.App:
#     try:
#         return firebase_admin.get_app()
#     except ValueError:
#         project_id = _resolve_firebase_project_id()
#         logger.info("Initializing Firebase Admin SDK for project %s", project_id)
#         try:
#             return firebase_admin.initialize_app(
#                 credentials.ApplicationDefault(),
#                 {"projectId": project_id},
#             )
#         except ValueError:
#             # Another thread/process initialized between get_app() and initialize_app()
#             return firebase_admin.get_app()
#         except Exception as exc:  # pragma: no cover - startup validation
#             logger.exception("Failed to initialize Firebase Admin SDK")
#             raise RuntimeError(
#                 "Could not initialize Firebase Admin SDK. Ensure application default credentials are configured."
#             ) from exc
        
def _initialize_firebase_app() -> firebase_admin.App:
    if not firebase_admin._apps:
        project_id = _resolve_firebase_project_id()
        logger.info("Initializing Firebase Admin SDK for project %s", project_id)
        default_app = firebase_admin.initialize_app(
        credentials.ApplicationDefault(),
                {"projectId": project_id},
    )
    return default_app




firebase_app = _initialize_firebase_app()

try:
    firestore_client = firestore.Client()
except Exception as exc:  # pragma: no cover - happens during startup misconfiguration
    logger.exception("Failed to initialize Firestore client")
    raise RuntimeError(
        "Could not initialize Firestore client. Ensure Google Cloud credentials are configured."
    ) from exc

ALLOWED_ORIGINS = _load_allowed_origins()


class AuthenticatedUser(TypedDict):
    uid: str
    email: str | None
    name: str | None


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be in the format 'Bearer <token>'.",
        )
    return token.strip()


def require_authenticated_user(
    authorization: str | None = Header(default=None),
) -> AuthenticatedUser:
    token = _extract_bearer_token(authorization)
    try:
        decoded = firebase_auth.verify_id_token(token, app=firebase_app)
    except firebase_auth.ExpiredIdTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please sign in again."
        ) from exc
    except firebase_auth.InvalidIdTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token."
        ) from exc
    except Exception as exc:  # pragma: no cover - firebase transient failures
        logger.exception("Unexpected error verifying Firebase ID token")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify authentication token.",
        ) from exc

    return AuthenticatedUser(
        uid=decoded["uid"],
        email=decoded.get("email"),
        name=decoded.get("name") or decoded.get("displayName"),
    )


app = FastAPI(title="Prep Pal Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            text_content = "\n".join(page.get_text("text") for page in document)
    except fitz.FileDataError as exc:  # type: ignore[attr-defined]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF is not a valid PDF file.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive catch, logged below
        logger.exception("Unexpected error parsing uploaded PDF")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse the uploaded PDF.",
        ) from exc

    cleaned = text_content.strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF has no extractable text.",
        )
    return cleaned


def extract_docx_text(docx_bytes: bytes) -> str:
    try:
        result = mammoth.extract_raw_text(io.BytesIO(docx_bytes))
        text_content = result.value or ""
    except Exception as exc:  # pragma: no cover - defensive conversion catch
        logger.exception("Unexpected error parsing uploaded DOCX")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse the uploaded DOCX file.",
        ) from exc

    cleaned = text_content.strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded DOCX has no extractable text.",
        )
    return cleaned


def detect_file_kind(filename: str | None, content_type: str | None, file_bytes: bytes) -> Literal["pdf", "docx"]:
    normalized_name = (filename or "").lower()
    extension = Path(normalized_name).suffix

    if (
        content_type in PDF_MIME_TYPES
        or extension == ".pdf"
        or file_bytes.startswith(b"%PDF")
    ):
        return "pdf"

    if (
        content_type in DOCX_MIME_TYPES
        or extension == ".docx"
        or file_bytes.startswith(b"PK")
    ):
        return "docx"

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported file type. Upload a PDF or DOCX file.",
    )


def normalize_study_text(raw_text: str) -> str:
    normalized_newlines = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_spaces = re.sub(r"[ \t]+", " ", normalized_newlines)
    normalized_paragraphs = re.sub(r"\n{3,}", "\n\n", normalized_spaces)
    return normalized_paragraphs.strip()


def sanitize_or_reject(raw_text: str) -> tuple[str, bool]:
    normalized = normalize_study_text(raw_text)
    lowered = normalized.lower()

    for pattern in STRICT_DENY_PATTERNS:
        if pattern.search(lowered):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study material contains prohibited language. Please clean it up and try again.",
            )

    sanitized = profanity.censor(normalized, censor_char="*")
    return sanitized, sanitized != normalized


async def generate_topic_prompt(pdf_text: str) -> tuple[str, str]:
    """Use Gemini to analyze PDF content and generate a study prompt"""
    try:

        client = genai.Client(vertexai=True, project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('GOOGLE_CLOUD_LOCATION'))

        analysis_prompt = f"""
        Analyze the following text from a PDF document and:
        1. Identify the main topic/subject
        2. Create a comprehensive study prompt that covers the key concepts, theories, and important details.
        Text content:
        {pdf_text}

        # Please respond in JSON format:
        # {{
        #     "topic": "Brief topic name",
        #     "study_prompt": "Detailed prompt covering key concepts for quiz generation"
        # }}
        """

        response = client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=analysis_prompt
        )

        try:
            # Parse the JSON response
            result = json.loads(response.text)
            return result["topic"], result["study_prompt"]
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            topic = "General Study Material"
            prompt = response.text
            return topic, prompt

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {e!s}") from e


def write_session_to_firestore(
    *,
    account_uid: str,
    room_name: str,
    user_id: str,
    user_name: str,
    grade: int,
    avatar: str,
    # topic: str,
    # study_prompt: str,
    study_metadata: dict[str, Any],
) -> None:
    document_data: dict[str, Any] = {
        "account_uid": account_uid,
        "room_name": room_name,
        "user_id": user_id,
        "user": {
            "name": user_name,
            "grade": grade,
            "avatar": avatar,
        },
        # "study_topic": topic,
        # "study_prompt": study_prompt,
        **study_metadata,
    }
    firestore_client.collection(SESSION_COLLECTION).document(room_name).set(document_data)



@app.post("/session-config", response_class=JSONResponse)
async def create_session_config(
    name: Annotated[str, Form(...)],
    grade: Annotated[str, Form(...)],
    avatar_id: Annotated[str, Form(...)],
    study_mode: Annotated[str, Form(...)],
    auth_user: Annotated[AuthenticatedUser, Depends(require_authenticated_user)],
    study_text: Annotated[str | None, Form()] = None,
    study_file: Annotated[UploadFile | None, File()] = None,
) -> JSONResponse:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name is required.")

    grade_value = grade.strip()
    if grade_value not in ALLOWED_GRADES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Grade must be between 1 and 12.")

    try:
        grade_int = int(grade_value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Grade must be numeric.") from exc

    avatar_choice = avatar_id.strip()
    if avatar_choice == "avatar1":
        avatar_choice = "avatar"
    if avatar_choice not in ALLOWED_AVATARS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid avatar selection.")

    requested_mode = study_mode.strip().lower()
    if requested_mode not in STUDY_INPUT_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Study mode must be either 'file' or 'text'.",
        )

    lesson_text: str | None = None
    lesson_source: dict[str, Any] = {"mode": requested_mode}

    if requested_mode == "text":
        if study_text is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study text is required when text mode is selected.",
            )

        trimmed = study_text.strip()
        if not trimmed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study text cannot be empty.",
            )
        if len(trimmed) > MAX_TEXT_LENGTH_CHARS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Study text cannot exceed {MAX_TEXT_LENGTH_CHARS:,} characters.",
            )

        sanitized_text, was_sanitized = sanitize_or_reject(trimmed)
        lesson_text = sanitized_text
        lesson_source.update(
            {
                "character_count": len(lesson_text),
                "was_sanitized": was_sanitized,
            }
        )
    else:
        if study_file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A file upload is required when file mode is selected.",
            )

        file_bytes = await study_file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds size limit of 10 MB.",
            )

        file_kind = detect_file_kind(study_file.filename, study_file.content_type, file_bytes)
        extracted_text = (
            extract_pdf_text(file_bytes)
            if file_kind == "pdf"
            else extract_docx_text(file_bytes)
        )
        sanitized_text, was_sanitized = sanitize_or_reject(extracted_text)
        lesson_text = sanitized_text
        lesson_source.update(
            {
                "file_kind": file_kind,
                "filename": study_file.filename or f"lesson.{file_kind}",
                "size_bytes": len(file_bytes),
                "character_count": len(lesson_text),
                "was_sanitized": was_sanitized,
            }
        )

    if not lesson_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No study material could be extracted.",
        )

    account_uid = auth_user["uid"]
    user_id = f"user_{account_uid}_{uuid4().hex[:8]}"
    room_name = f"study_room_{uuid4().hex[:12]}"
    participant_identity = f"student_{user_id}"

    study_metadata = {
        "lesson_text": lesson_text,
        "lesson_source": lesson_source,
        # Maintain backwards compatibility while the agent migrates off pdf_text
        "pdf_text": lesson_text,
    }

    # topic, study_prompt = await generate_topic_prompt(pdf_text)

    try:
        write_session_to_firestore(
            account_uid=account_uid,
            room_name=room_name,
            user_id=user_id,
            user_name=cleaned_name,
            grade=grade_int,
            avatar=avatar_choice,
            # topic=topic,
            # study_prompt=study_prompt,
            study_metadata=study_metadata,
        )
    except Exception as exc:  # pragma: no cover - external service failure
        logger.exception("Failed to write session config to Firestore")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to persist session configuration.",
        ) from exc

    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LiveKit configuration is incomplete on the server.",
        )

    token = (
        lk_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(participant_identity)
        .with_name(cleaned_name or participant_identity)
        .with_metadata(
            json.dumps(
                {
                    "user_id": user_id,
                    "account_uid": account_uid,
                    "email": auth_user.get("email"),
                }
            )
        )
        .with_grants(
            lk_api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
    )

    participant_token = token.to_jwt()

    logger.info("Stored session config for user %s in room %s", user_id, room_name)

    return JSONResponse(
        {
            "user_id": user_id,
            "connection": {
                "serverUrl": LIVEKIT_URL,
                "roomName": room_name,
                "participantName": participant_identity,
                "participantToken": participant_token,
            },
        }
    )
