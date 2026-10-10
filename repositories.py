"""Parameterized repositories for Memora's PostgreSQL foundation.

Every repository accepts an optional connection factory.  Tests can inject a
fake connection, while production callers use :func:`database.get_connection`.
Repository methods open their own transaction by default, or can share a
caller-provided connection for a larger unit of work.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, Mapping, Sequence

from database import get_connection


ConnectionFactory = Callable[[], Any]


def _json_payload(value: Any, default: Any) -> str:
    """Serialize JSON values before binding them to a ``::jsonb`` parameter."""

    return json.dumps(
        default if value is None else value,
        ensure_ascii=False,
        allow_nan=False,
    )


def _vector_literal(embedding: Sequence[float]) -> str:
    """Create a safe pgvector literal from numeric values.

    Values are converted to floats before joining, so arbitrary strings cannot
    become SQL.  Dimensions are checked locally as well as by ``vector(384)``
    in PostgreSQL, making invalid embeddings testable without a database.
    """

    if isinstance(embedding, (str, bytes)):
        raise TypeError("embedding must be a sequence of numbers")

    try:
        values = list(embedding)
    except TypeError as error:
        raise TypeError("embedding must be a sequence of numbers") from error

    if len(values) != 384:
        raise ValueError("embedding must contain exactly 384 values")

    normalized: list[str] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite numbers")
        normalized.append(format(number, ".17g"))
    return "[" + ",".join(normalized) + "]"


def _page_values(limit: int, offset: int = 0) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    return limit, offset


@contextmanager
def _cursor(connection: Any) -> Iterator[Any]:
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()


class Repository:
    """Base class providing injectable and transaction-safe connections."""

    def __init__(
        self,
        connection_factory: ConnectionFactory | Any | None = None,
        *,
        connection: Any | None = None,
    ):
        # A direct connection is convenient for unit tests.  A callable remains
        # a factory so production methods can open and close one transaction per
        # operation.
        if connection is not None and connection_factory is not None:
            raise ValueError("pass either connection or connection_factory, not both")
        if connection is None and connection_factory is not None and not callable(
            connection_factory
        ):
            connection = connection_factory
            connection_factory = None
        self._bound_connection = connection
        self._connection_factory = connection_factory or get_connection

    @contextmanager
    def _connection_scope(self, connection: Any | None) -> Iterator[Any]:
        if connection is None:
            connection = self._bound_connection
        if connection is not None:
            with nullcontext(connection) as existing:
                yield existing
            return

        opened = self._connection_factory()
        if hasattr(opened, "__enter__") and hasattr(opened, "__exit__"):
            with opened as managed:
                yield managed
        else:
            with nullcontext(opened) as managed:
                yield managed


class VideoRepository(Repository):
    def create(
        self,
        title: str,
        source_url: str | None = None,
        file_path: str | None = None,
        status: str = "pending",
        metadata: Mapping[str, Any] | None = None,
        video_id: str | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        sql = """
            insert into public.videos
                (id, title, source_url, file_path, status, metadata)
            values
                (coalesce(%s::uuid, gen_random_uuid()), %s, %s, %s, %s, %s::jsonb)
            returning *
        """
        params = (
            video_id,
            title,
            source_url,
            file_path,
            status,
            _json_payload(metadata, {}),
        )
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def get(self, video_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                "select * from public.videos where id = %s::uuid",
                (video_id,),
            )
            return cursor.fetchone()

    def list(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        sql = """
            select *
            from public.videos
            where (%s::text is null or status = %s)
            order by created_at desc
            limit %s offset %s
        """
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(sql, (status, status, limit, offset))
            return list(cursor.fetchall())

    def update_status(
        self,
        video_id: str,
        status: str,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                update public.videos
                set status = %s
                where id = %s::uuid
                returning *
                """,
                (status, video_id),
            )
            return cursor.fetchone()


class TranscriptChunkRepository(Repository):
    def create(
        self,
        video_id: str,
        chunk_index: int,
        content: str,
        start_seconds: float,
        end_seconds: float,
        embedding: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        sql = """
            insert into public.transcript_chunks
                (video_id, chunk_index, content, start_seconds, end_seconds,
                 embedding, metadata)
            values
                (%s::uuid, %s, %s, %s, %s, %s::vector, %s::jsonb)
            returning *
        """
        params = (
            video_id,
            chunk_index,
            content,
            start_seconds,
            end_seconds,
            _vector_literal(embedding),
            _json_payload(metadata, {}),
        )
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def get(self, chunk_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                "select * from public.transcript_chunks where id = %s::uuid",
                (chunk_id,),
            )
            return cursor.fetchone()

    def list_for_video(
        self,
        video_id: str,
        limit: int = 1000,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.transcript_chunks
                where video_id = %s::uuid
                order by chunk_index
                limit %s offset %s
                """,
                (video_id, limit, offset),
            )
            return list(cursor.fetchall())

    def search(
        self,
        embedding: Sequence[float],
        match_threshold: float = 0.0,
        match_count: int = 5,
        video_id: str | None = None,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        if isinstance(match_count, bool) or not isinstance(match_count, int):
            raise ValueError("match_count must be an integer")
        if match_count < 1:
            raise ValueError("match_count must be positive")
        threshold = float(match_threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("match_threshold must be a finite number between -1 and 1")
        sql = """
            select *
            from public.match_transcript_chunks(
                %s::vector, %s::real, %s::integer, %s::uuid
            )
        """
        params = (
            _vector_literal(embedding),
            threshold,
            match_count,
            video_id,
        )
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    match = search


class ProcessingJobRepository(Repository):
    def create(
        self,
        video_id: str,
        job_type: str = "transcription",
        metadata: Mapping[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                insert into public.processing_jobs (video_id, job_type, metadata)
                values (%s::uuid, %s, %s::jsonb)
                returning *
                """,
                (video_id, job_type, _json_payload(metadata, {})),
            )
            return cursor.fetchone()

    def get(self, job_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                "select * from public.processing_jobs where id = %s::uuid",
                (job_id,),
            )
            return cursor.fetchone()

    def list_for_video(
        self,
        video_id: str,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.processing_jobs
                where video_id = %s::uuid
                  and (%s::text is null or status = %s)
                order by created_at desc
                limit %s offset %s
                """,
                (video_id, status, status, limit, offset),
            )
            return list(cursor.fetchall())

    def claim(self, job_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                update public.processing_jobs
                set status = 'processing',
                    started_at = coalesce(started_at, now()),
                    attempts = attempts + 1,
                    error_message = null,
                    finished_at = null
                where id = %s::uuid
                  and status in ('queued', 'failed')
                returning *
                """,
                (job_id,),
            )
            return cursor.fetchone()

    def update_status(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                update public.processing_jobs
                set status = %s,
                    error_message = %s,
                    finished_at = case
                        when %s::text in ('completed', 'failed') then now()
                        else null
                    end
                where id = %s::uuid
                returning *
                """,
                (status, error_message, status, job_id),
            )
            return cursor.fetchone()


class ChatRepository(Repository):
    def create_session(
        self,
        title: str | None = None,
        video_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                insert into public.chat_sessions (video_id, title, metadata)
                values (%s::uuid, %s, %s::jsonb)
                returning *
                """,
                (video_id, title, _json_payload(metadata, {})),
            )
            return cursor.fetchone()

    def get_session(self, session_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                "select * from public.chat_sessions where id = %s::uuid",
                (session_id,),
            )
            return cursor.fetchone()

    def list_sessions(
        self,
        video_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.chat_sessions
                where (%s::uuid is null or video_id = %s::uuid)
                order by updated_at desc
                limit %s offset %s
                """,
                (video_id, video_id, limit, offset),
            )
            return list(cursor.fetchall())

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Any = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                insert into public.chat_messages
                    (session_id, role, content, citations, metadata)
                values (%s::uuid, %s, %s, %s::jsonb, %s::jsonb)
                returning *
                """,
                (
                    session_id,
                    role,
                    content,
                    _json_payload(citations, []),
                    _json_payload(metadata, {}),
                ),
            )
            message = cursor.fetchone()
            cursor.execute(
                "update public.chat_sessions set updated_at = now() where id = %s::uuid",
                (session_id,),
            )
            return message

    def get_message(self, message_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                "select * from public.chat_messages where id = %s::uuid",
                (message_id,),
            )
            return cursor.fetchone()

    def list_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.chat_messages
                where session_id = %s::uuid
                order by created_at
                limit %s offset %s
                """,
                (session_id, limit, offset),
            )
            return list(cursor.fetchall())


class QuizRepository(Repository):
    def create_quiz(
        self,
        title: str | None = None,
        topic: str | None = None,
        video_id: str | None = None,
        session_id: str | None = None,
        status: str = "draft",
        metadata: Mapping[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                insert into public.quizzes
                    (video_id, session_id, title, topic, status, metadata)
                values (%s::uuid, %s::uuid, %s, %s, %s, %s::jsonb)
                returning *
                """,
                (
                    video_id,
                    session_id,
                    title,
                    topic,
                    status,
                    _json_payload(metadata, {}),
                ),
            )
            return cursor.fetchone()

    create = create_quiz

    def get(self, quiz_id: str, *, connection: Any | None = None) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                "select * from public.quizzes where id = %s::uuid",
                (quiz_id,),
            )
            return cursor.fetchone()

    def list(
        self,
        video_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.quizzes
                where (%s::uuid is null or video_id = %s::uuid)
                  and (%s::uuid is null or session_id = %s::uuid)
                order by created_at desc
                limit %s offset %s
                """,
                (video_id, video_id, session_id, session_id, limit, offset),
            )
            return list(cursor.fetchall())

    def add_question(
        self,
        quiz_id: str,
        question_number: int,
        prompt: str,
        options: Mapping[str, Any],
        correct_option: str | None = None,
        explanation: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                insert into public.quiz_questions
                    (quiz_id, question_number, prompt, options, correct_option,
                     explanation, source, metadata)
                values (%s::uuid, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
                returning *
                """,
                (
                    quiz_id,
                    question_number,
                    prompt,
                    _json_payload(options, {}),
                    correct_option,
                    explanation,
                    source,
                    _json_payload(metadata, {}),
                ),
            )
            return cursor.fetchone()

    create_question = add_question

    def list_questions(
        self,
        quiz_id: str,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.quiz_questions
                where quiz_id = %s::uuid
                order by question_number
                limit %s offset %s
                """,
                (quiz_id, limit, offset),
            )
            return list(cursor.fetchall())

    def create_attempt(
        self,
        quiz_id: str,
        answers: Mapping[str, Any],
        score: float | None = None,
        correct_count: int | None = None,
        total_questions: int | None = None,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        submitted_at: Any | None = None,
        *,
        connection: Any | None = None,
    ) -> Any:
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                insert into public.quiz_attempts
                    (quiz_id, chat_session_id, answers, score, correct_count,
                     total_questions, submitted_at, metadata)
                values (%s::uuid, %s::uuid, %s::jsonb, %s, %s, %s, %s, %s::jsonb)
                returning *
                """,
                (
                    quiz_id,
                    session_id,
                    _json_payload(answers, {}),
                    score,
                    correct_count,
                    total_questions,
                    submitted_at,
                    _json_payload(metadata, {}),
                ),
            )
            return cursor.fetchone()

    record_attempt = create_attempt

    def list_attempts(
        self,
        quiz_id: str,
        limit: int = 100,
        offset: int = 0,
        *,
        connection: Any | None = None,
    ) -> list[Any]:
        limit, offset = _page_values(limit, offset)
        with self._connection_scope(connection) as conn, _cursor(conn) as cursor:
            cursor.execute(
                """
                select *
                from public.quiz_attempts
                where quiz_id = %s::uuid
                order by created_at desc
                limit %s offset %s
                """,
                (quiz_id, limit, offset),
            )
            return list(cursor.fetchall())


class ChatSessionRepository(ChatRepository):
    """Focused session API for callers that do not need message helpers."""

    create = ChatRepository.create_session
    get = ChatRepository.get_session
    list = ChatRepository.list_sessions


class ChatMessageRepository(ChatRepository):
    """Focused message API for callers that do not need session helpers."""

    create = ChatRepository.add_message
    get = ChatRepository.get_message
    list = ChatRepository.list_messages


# Short aliases keep the foundation convenient without duplicating SQL.
ChunkRepository = TranscriptChunkRepository
JobRepository = ProcessingJobRepository


__all__ = [
    "ChatMessageRepository",
    "ChatRepository",
    "ChatSessionRepository",
    "ChunkRepository",
    "JobRepository",
    "ProcessingJobRepository",
    "QuizRepository",
    "Repository",
    "TranscriptChunkRepository",
    "VideoRepository",
]
