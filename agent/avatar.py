import logging
import json
import uuid
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List, TypedDict
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, RoomOutputOptions
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import hedra, google
from google.cloud import firestore
import os
from PIL import Image

load_dotenv()

logger = logging.getLogger("avatar")
logger.setLevel(logging.INFO)

SESSION_COLLECTION = "session_configs"

try:
    firestore_client = firestore.Client()
except Exception as exc:  # pragma: no cover - startup validation
    logger.exception("Failed to initialize Firestore client for avatar agent")
    raise

class QuizAnswerDict(TypedDict):
    text: str
    is_correct: bool

class QuizQuestionDict(TypedDict):
    text: str
    answers: List[QuizAnswerDict]

@dataclass
class FlashCard:
    """Class to represent a flash card."""
    id: str
    question: str
    answer: str
    is_flipped: bool = False

@dataclass
class QuizAnswer:
    """Class to represent a quiz answer option."""
    id: str
    text: str
    is_correct: bool

@dataclass
class QuizQuestion:
    """Class to represent a quiz question."""
    id: str
    text: str
    answers: List[QuizAnswer]

@dataclass
class Quiz:
    """Class to represent a quiz."""
    id: str
    questions: List[QuizQuestion]

@dataclass
class UserData:
    """Class to store user data during a session."""
    ctx: Optional[JobContext] = None
    user_id: Optional[str] = None
    user_profile: Dict[str, Any] = field(default_factory=dict)
    lesson_text: Optional[str] = None
    pdf_text: Optional[str] = None
    flash_cards: List[FlashCard] = field(default_factory=list)
    quizzes: List[Quiz] = field(default_factory=list)

    def reset(self) -> None:
        """Reset session data."""
        # Keep flash cards and quizzes intact
    def add_flash_card(self, question: str, answer: str) -> FlashCard:
        """Add a new flash card to the collection."""
        card = FlashCard(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer
        )
        self.flash_cards.append(card)
        return card

    def get_flash_card(self, card_id: str) -> Optional[FlashCard]:
        """Get a flash card by ID."""
        for card in self.flash_cards:
            if card.id == card_id:
                return card
        return None

    def flip_flash_card(self, card_id: str) -> Optional[FlashCard]:
        """Flip a flash card by ID."""
        card = self.get_flash_card(card_id)
        if card:
            card.is_flipped = not card.is_flipped
            return card
        return None

    def add_quiz(self, questions: List[QuizQuestionDict]) -> Quiz:
        """Add a new quiz to the collection."""
        quiz_questions = []
        for q in questions:
            answers = []
            for a in q["answers"]:
                answers.append(QuizAnswer(
                    id=str(uuid.uuid4()),
                    text=a["text"],
                    is_correct=a["is_correct"]
                ))
            quiz_questions.append(QuizQuestion(
                id=str(uuid.uuid4()),
                text=q["text"],
                answers=answers
            ))

        quiz = Quiz(
            id=str(uuid.uuid4()),
            questions=quiz_questions
        )
        self.quizzes.append(quiz)
        return quiz

    def get_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Get a quiz by ID."""
        for quiz in self.quizzes:
            if quiz.id == quiz_id:
                return quiz
        return None

    def check_quiz_answers(self, quiz_id: str, user_answers: dict) -> List[tuple]:
        """Check user's quiz answers and return results."""
        quiz = self.get_quiz(quiz_id)
        if not quiz:
            return []

        results = []
        for question in quiz.questions:
            user_answer_id = user_answers.get(question.id)

            # Find the selected answer and the correct answer
            selected_answer = None
            correct_answer = None

            for answer in question.answers:
                if answer.id == user_answer_id:
                    selected_answer = answer
                if answer.is_correct:
                    correct_answer = answer

            is_correct = selected_answer and selected_answer.is_correct
            results.append((question, selected_answer, correct_answer, is_correct))

        return results

class AvatarAsset(TypedDict):
    image: str
    voice: str


AVATAR_IMAGE_FILES: Dict[str, AvatarAsset] = {
    "avatar1": {"image": "avatar1.png", "voice": "Enceladus"},
    "avatar2": {"image": "avatar2.png", "voice": "Orus"},
    "avatar3": {"image": "avatar3.png", "voice": "Kore"},
    "avatar4": {"image": "avatar4.png", "voice": "Puck"},
}

DEFAULT_AVATAR_ASSET = AVATAR_IMAGE_FILES["avatar1"]
DEFAULT_AVATAR_IMAGE = "avatar1.png"
DEFAULT_AVATAR_VOICE = "Enceladus"

BASE_AGENT_INSTRUCTIONS = textwrap.dedent(
    """
                 **CORE DIRECTIVE: YOU ARE A TOOL-USING AI.**
                You are a helpful, patient, and curious study partner.
                Your primary goal is to foster deep understanding through guided discovery, dialogue, and repetition by using the function calls and tools provided to help the user.

                Your responsibilities include:
                    •\tExplaining core topics found in the student's current study material.
                    •\tUsing Socratic questioning to help the student reach answers themselves.
                    •\tProviding clear explanations only when necessary, then reinforcing them with follow-up questions.
                    •\tAlternating between roles: sometimes you ask questions, other times you answer.
                    •\tMaintaining a friendly, encouraging tone that supports confidence and curiosity.

                Do not rush to give the answer. Instead, support reasoning, analytical thinking, and the development of historical understanding.

                Because you are speaking via voice, use clear, descriptive language and reference concrete details (names, dates, events, definitions) drawn from the provided study material.

                Always start answering a question by first asking questions—lean into the Socratic method until the student is ready for direct explanations.

                === AVAILABLE FUNCTIONS ===
                FLASH CARDS FUNCTION:
                You can create flash cards to help the user learn and remember important concepts. Use the create_flash_card function
                to create a new flash card with a question and answer. The flash card will appear beside you in the UI.

                Be proactive in creating flash cards for important concepts, especially when:
                - Teaching new vocabulary or terminology
                - Explaining complex principles that are worth remembering
                - Summarizing key points from a discussion

                Do not tell the user the answer before they look at it!
                You can also flip flash cards to show the answer using the flip_flash_card function.

                QUIZ FUNCTION:
                You can create multiple-choice quizzes to test the user's knowledge. Use the create_quiz function
                to create a new quiz with questions and multiple-choice answers. The quiz will appear on the left side of the UI.

                For each question, you should provide:
                - A clear question text
                - 3-5 answer options (one must be marked as correct)

                Quizzes are great for:
                - Testing comprehension after explaining a concept
                - Reviewing previously covered material
                - Preparing the user for a test or exam
                - Breaking up longer learning sessions with interactive elements

                When the user submits their answers, provide verbal feedback that includes interesting supporting details pulled from the lesson.
                For any incorrectly answered questions, create flash cards that help them study the correct information.

                Start the interaction with a short introduction, and let the student
                guide their own learning journey!

                Keep your speaking turns short, only one or two sentences. We want the
                student to do most of the speaking.
            """
)


def compose_agent_instructions(
    *,
    grade: Optional[str],
    lesson_text: Optional[str],
    extra_instructions: Optional[str],
) -> str:
    base = BASE_AGENT_INSTRUCTIONS.strip()

    prep_lines: List[str] = []
    if grade:
        prep_lines.append(f"- The student is in grade {grade}. Match explanations to that level.")
    else:
        prep_lines.append("- Grade level unknown. Ask early to gauge difficulty and adjust accordingly.")

    material_text = (lesson_text or "").strip()
    if material_text:
        prep_lines.append("- Lesson text to reference:\n" + material_text)
    else:
        prep_lines.append("- No lesson text provided. Encourage the student to summarize their study guide.")

    prep_block = "Session Preparation:\n" + "\n".join(prep_lines)
    instructions = f"{base}\n\n{prep_block}" if prep_lines else base

    if extra_instructions:
        instructions = f"{instructions}\n\n{extra_instructions.strip()}"

    return instructions


def load_session_context(room_name: str) -> Dict[str, Any]:
    session_ref = firestore_client.collection(SESSION_COLLECTION).document(room_name)
    session_doc = session_ref.get()
    if not session_doc.exists:
        raise ValueError(f"Session configuration for room '{room_name}' not found")

    session_data = session_doc.to_dict() or {}
    if "user_id" not in session_data:
        raise ValueError(f"Session document for room '{room_name}' missing user_id")
    return session_data


def build_dynamic_instructions(session_data: Dict[str, Any]) -> str:
    user_info = session_data.get("user", {})
    lines: List[str] = ["Student Context:"]

    name = user_info.get("name")
    grade = user_info.get("grade")
    if name:
        lines.append(f"- Name: {name}")

    if grade is not None:
        lines.append(f"- Grade: {grade}")

    avatar_choice = user_info.get("avatar")
    if avatar_choice:
        lines.append(f"- Preferred Avatar: {avatar_choice}")

    study_prompt = session_data.get("study_prompt")
    prompt_section = ""
    if study_prompt:
        prompt_section = textwrap.dedent(
            f"""

            Study Prompt Guidance:
            {study_prompt}
            """
        ).strip()

    context_block = "\n".join(lines)
    if prompt_section:
        return f"{context_block}\n\n{prompt_section}"
    return context_block



class AvatarAgent(Agent):
    def __init__(
        self,
        *,
        grade: Optional[str] = None,
        lesson_text: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> None:
        instructions = compose_agent_instructions(
            grade=grade,
            lesson_text=lesson_text,
            extra_instructions=extra_instructions,
        )
        super().__init__(instructions=instructions)


    @function_tool
    async def create_flash_card(self, context: RunContext[UserData], question: str, answer: str):
        """Create a new flash card and display it to the user.

        Args:
            question: The question or front side of the flash card
            answer: The answer or back side of the flash card
        """
        userdata = context.userdata
        card = userdata.add_flash_card(question, answer)

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return f"Created a flash card, but couldn't access the room to send it."

        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return f"Created a flash card, but no participants found to send it to."

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return f"Created a flash card, but couldn't get the first participant."
        payload = {
            "action": "show",
            "id": card.id,
            "question": card.question,
            "answer": card.answer,
            "index": len(userdata.flash_cards) - 1
        }

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending flash card payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.flashcard",
            payload=json_payload
        )

        return f"I've created a flash card with the question: '{question}'"

    @function_tool
    async def flip_flash_card(self, context: RunContext[UserData], card_id: str):
        """Flip a flash card to show the answer or question.

        Args:
            card_id: The ID of the flash card to flip
        """
        userdata = context.userdata
        card = userdata.flip_flash_card(card_id)

        if not card:
            return f"Flash card with ID {card_id} not found."

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return f"Flipped the flash card, but couldn't access the room to send it."

        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return f"Flipped the flash card, but no participants found to send it to."

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return f"Flipped the flash card, but couldn't get the first participant."
        payload = {
            "action": "flip",
            "id": card.id
        }

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending flip card payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.flashcard",
            payload=json_payload
        )

        return f"I've flipped the flash card to show the {'answer' if card.is_flipped else 'question'}"

    @function_tool
    async def create_quiz(self, context: RunContext[UserData], questions: List[QuizQuestionDict]):
        """Create a new quiz with multiple choice questions and display it to the user.

        Args:
            questions: A list of question objects. Each question object should have:
                - text: The question text
                - answers: A list of answer objects, each with:
                    - text: The answer text
                    - is_correct: Boolean indicating if this is the correct answer
        """
        userdata = context.userdata
        quiz = userdata.add_quiz(questions)

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return f"Created a quiz, but couldn't access the room to send it."

        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return f"Created a quiz, but no participants found to send it to."

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return f"Created a quiz, but couldn't get the first participant."

        # Format questions for client
        client_questions = []
        for q in quiz.questions:
            client_answers = []
            for a in q.answers:
                client_answers.append({
                    "id": a.id,
                    "text": a.text
                })
            client_questions.append({
                "id": q.id,
                "text": q.text,
                "answers": client_answers
            })

        payload = {
            "action": "show",
            "id": quiz.id,
            "questions": client_questions
        }

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending quiz payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.quiz",
            payload=json_payload
        )

        return f"I've created a quiz with {len(questions)} questions. Please answer them when you're ready."

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session_context: Dict[str, Any] = {}
    extra_instructions: Optional[str] = None

    try:
        room_name = ctx.room.name if ctx.room else ""
        if room_name:
            session_context = load_session_context(room_name)
            extra_instructions = build_dynamic_instructions(session_context)
            logger.info("Loaded session context for room %s", room_name)
        else:
            logger.warning("Job context provided no room name; proceeding with default instructions")
    except Exception as exc:  # pragma: no cover - defensive logging for missing context
        logger.exception("Failed to load session configuration prior to agent start: %s", exc)

    user_profile: Dict[str, Any] = session_context.get("user", {}) if session_context else {}
    grade_value = user_profile.get("grade")
    lesson_text = None
    if session_context:
        lesson_text = session_context.get("lesson_text") or session_context.get("pdf_text")

    agent = AvatarAgent(
        grade=str(grade_value) if grade_value is not None else None,
        lesson_text=lesson_text,
        extra_instructions=extra_instructions,
    )

    userdata = UserData(ctx=ctx)
    if session_context:
        userdata.user_id = session_context.get("user_id")
        userdata.user_profile = user_profile
        userdata.study_topic = session_context.get("study_topic")
        userdata.study_prompt = session_context.get("study_prompt")
        userdata.lesson_text = lesson_text
        userdata.pdf_text = lesson_text

    selected_avatar_key = (userdata.user_profile or {}).get("avatar")
    avatar_asset = AVATAR_IMAGE_FILES.get(selected_avatar_key, DEFAULT_AVATAR_ASSET)
    selected_voice = avatar_asset.get("voice", DEFAULT_AVATAR_VOICE)

    session = AgentSession[UserData](
        userdata=userdata,
        llm=google.beta.realtime.RealtimeModel(
            voice=selected_voice,
            model="gemini-2.0-flash-live-preview-04-09",
            vertexai=True,
            temperature=0.9,
            ),
        resume_false_interruption=False,
    )

    # Load the avatar image that matches the student's selection
    avatar_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_filename = avatar_asset.get("image", DEFAULT_AVATAR_IMAGE)
    avatar_image_path = os.path.join(avatar_dir, avatar_filename)

    if not os.path.exists(avatar_image_path):
        logger.warning(
            "Avatar image '%s' not found. Falling back to default avatar.",
            avatar_filename,
        )
        avatar_image_path = os.path.join(avatar_dir, DEFAULT_AVATAR_IMAGE)

    if not os.path.exists(avatar_image_path):
        raise FileNotFoundError(f"Avatar image not found at {avatar_image_path}")

    avatar_image = Image.open(avatar_image_path)
    logger.info(
        "Loaded avatar image '%s' for user '%s'",
        os.path.basename(avatar_image_path),
        userdata.user_id,
    )
    
    # Create the avatar session
    avatar = hedra.AvatarSession(
        avatar_participant_identity="education-avatar",
        avatar_image=avatar_image
    )

    # Register RPC method for flipping flash cards from client
    async def handle_flip_flash_card(rpc_data):
        try:
            logger.info(f"Received flash card flip payload: {rpc_data}")

            # Extract the payload from the RpcInvocationData object
            payload_str = rpc_data.payload
            logger.info(f"Extracted payload string: {payload_str}")

            # Parse the JSON payload
            payload_data = json.loads(payload_str)
            logger.info(f"Parsed payload data: {payload_data}")

            card_id = payload_data.get("id")

            if card_id:
                card = userdata.flip_flash_card(card_id)
                if card:
                    logger.info(f"Flipped flash card {card_id}, is_flipped: {card.is_flipped}")
                    # Send a message to the user via the agent, we're disabling this for now.
                    # session.generate_reply(user_input=(f"Please describe the {'answer' if card.is_flipped else 'question'}"))
                else:
                    logger.error(f"Card with ID {card_id} not found")
            else:
                logger.error("No card ID found in payload")

            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for payload '{rpc_data.payload}': {e}")
            return f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Error handling flip flash card: {e}")
            return f"error: {str(e)}"

    # Register RPC method for handling quiz submissions
    async def handle_submit_quiz(rpc_data):
        try:
            logger.info(f"Received quiz submission payload: {rpc_data}")

            # Extract the payload from the RpcInvocationData object
            payload_str = rpc_data.payload
            logger.info(f"Extracted quiz submission string: {payload_str}")

            # Parse the JSON payload
            payload_data = json.loads(payload_str)
            logger.info(f"Parsed quiz submission data: {payload_data}")

            quiz_id = payload_data.get("id")
            user_answers = payload_data.get("answers", {})

            if not quiz_id:
                logger.error("No quiz ID found in payload")
                return "error: No quiz ID found in payload"

            # Check the quiz answers
            quiz_results = userdata.check_quiz_answers(quiz_id, user_answers)
            if not quiz_results:
                logger.error(f"Quiz with ID {quiz_id} not found")
                return "error: Quiz not found"

            # Count correct answers
            correct_count = sum(1 for _, _, _, is_correct in quiz_results if is_correct)
            total_count = len(quiz_results)

            # Create a verbal response for the agent to say
            result_summary = f"You got {correct_count} out of {total_count} questions correct."

            # Generate feedback for each question
            feedback_details = []
            for question, selected_answer, correct_answer, is_correct in quiz_results:
                if is_correct:
                    feedback = f"Question: {question.text}\nYour answer: {selected_answer.text} ✓ Correct!"
                else:
                    feedback = f"Question: {question.text}\nYour answer: {selected_answer.text if selected_answer else 'None'} ✗ Incorrect. The correct answer is: {correct_answer.text}"

                    # Create a flash card for incorrectly answered questions
                    card = userdata.add_flash_card(question.text, correct_answer.text)
                    participant = next(iter(ctx.room.remote_participants.values()), None)
                    if participant:
                        flash_payload = {
                            "action": "show",
                            "id": card.id,
                            "question": card.question,
                            "answer": card.answer,
                            "index": len(userdata.flash_cards) - 1
                        }
                        json_flash_payload = json.dumps(flash_payload)
                        await ctx.room.local_participant.perform_rpc(
                            destination_identity=participant.identity,
                            method="client.flashcard",
                            payload=json_flash_payload
                        )

                feedback_details.append(feedback)

            detailed_feedback = "\n\n".join(feedback_details)
            full_response = f"{result_summary}\n\n{detailed_feedback}"

            # Have the agent say the results
            session.say(full_response)

            return "success"
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for quiz submission payload '{rpc_data.payload}': {e}")
            return f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Error handling quiz submission: {e}")
            return f"error: {str(e)}"

    # Register RPC methods - The method names need to match exactly what the client is calling
    logger.info("Registering RPC methods")
    ctx.room.local_participant.register_rpc_method(
        "agent.flipFlashCard",
        handle_flip_flash_card
    )

    ctx.room.local_participant.register_rpc_method(
        "agent.submitQuiz",
        handle_submit_quiz
    )

    # Start the avatar with the same session that has userdata
    await avatar.start(session, room=ctx.room)

    # Start the agent session with the same session object
    await session.start(
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            audio_enabled=True,  # Enable audio since we want the avatar to speak
        ),
        agent=agent
    )

    session.generate_reply(instructions="say hello to the user")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )
