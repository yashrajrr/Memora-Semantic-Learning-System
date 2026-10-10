-- Memora database foundation for Supabase PostgreSQL.
-- The migration is intentionally independent from the existing file-backed app
-- flow.  It can be applied by the Supabase migration tooling or database.py.

-- Supabase commonly installs extensions in the extensions schema.  Local
-- PostgreSQL installations may install them in public; keeping both schemas in
-- the search path makes the migration work in either case.
set search_path = public, extensions;

create extension if not exists vector with schema public;
create extension if not exists pgcrypto with schema public;

create table if not exists public.videos (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid references auth.users(id) on delete set null,
    title text not null check (length(btrim(title)) > 0),
    source_type text not null default 'upload'
        check (source_type in ('upload', 'youtube', 'external')),
    source_url text,
    file_path text,
    duration_seconds double precision
        check (duration_seconds is null or duration_seconds >= 0),
    status text not null default 'pending'
        check (status in ('pending', 'processing', 'ready', 'failed')),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.transcript_chunks (
    id uuid primary key default gen_random_uuid(),
    video_id uuid not null references public.videos(id) on delete cascade,
    chunk_index integer not null check (chunk_index >= 0),
    content text not null check (length(btrim(content)) > 0),
    start_seconds double precision not null
        check (start_seconds >= 0 and start_seconds < 'Infinity'::double precision),
    end_seconds double precision not null
        check (end_seconds >= start_seconds and end_seconds < 'Infinity'::double precision),
    embedding vector(384) not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (video_id, chunk_index)
);

create table if not exists public.processing_jobs (
    id uuid primary key default gen_random_uuid(),
    video_id uuid not null references public.videos(id) on delete cascade,
    job_type text not null check (length(btrim(job_type)) > 0),
    status text not null default 'queued'
        check (status in ('queued', 'processing', 'completed', 'failed')),
    attempts integer not null default 0 check (attempts >= 0),
    error_message text,
    metadata jsonb not null default '{}'::jsonb,
    started_at timestamptz,
    finished_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (
        (status in ('completed', 'failed') and finished_at is not null)
        or (status in ('queued', 'processing') and finished_at is null)
    )
);

create table if not exists public.chat_sessions (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid references auth.users(id) on delete set null,
    video_id uuid references public.videos(id) on delete set null,
    title text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references public.chat_sessions(id) on delete cascade,
    role text not null check (role in ('system', 'user', 'assistant')),
    content text not null check (length(btrim(content)) > 0),
    citations jsonb not null default '[]'::jsonb
        check (jsonb_typeof(citations) = 'array'),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.quizzes (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid references auth.users(id) on delete set null,
    video_id uuid references public.videos(id) on delete set null,
    session_id uuid references public.chat_sessions(id) on delete set null,
    title text,
    topic text,
    status text not null default 'draft'
        check (status in ('draft', 'ready', 'completed', 'failed')),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.quiz_questions (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid not null references public.quizzes(id) on delete cascade,
    question_number integer not null check (question_number > 0),
    prompt text not null check (length(btrim(prompt)) > 0),
    options jsonb not null check (jsonb_typeof(options) = 'object'),
    correct_option text check (correct_option is null or options ? correct_option),
    explanation text,
    source text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (quiz_id, question_number)
);

create table if not exists public.quiz_attempts (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid not null references public.quizzes(id) on delete cascade,
    chat_session_id uuid references public.chat_sessions(id) on delete set null,
    answers jsonb not null default '{}'::jsonb
        check (jsonb_typeof(answers) = 'object'),
    score numeric(6, 2) check (score >= 0 and score <= 100),
    correct_count integer check (correct_count >= 0),
    total_questions integer check (total_questions >= 0),
    submitted_at timestamptz,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (
        correct_count is null
        or total_questions is null
        or correct_count <= total_questions
    )
);

create index if not exists videos_status_idx
    on public.videos (status);
create index if not exists videos_owner_id_created_at_idx
    on public.videos (owner_id, created_at desc);
create index if not exists videos_created_at_idx
    on public.videos (created_at desc);

-- The unique (video_id, chunk_index) constraint already supplies its B-tree index.
create index if not exists transcript_chunks_embedding_hnsw_idx
    on public.transcript_chunks
    using hnsw (embedding vector_cosine_ops);

create index if not exists processing_jobs_status_created_at_idx
    on public.processing_jobs (status, created_at);
create index if not exists processing_jobs_video_id_idx
    on public.processing_jobs (video_id, created_at desc);

create index if not exists chat_sessions_video_id_updated_at_idx
    on public.chat_sessions (video_id, updated_at desc);
create index if not exists chat_sessions_owner_id_updated_at_idx
    on public.chat_sessions (owner_id, updated_at desc);
create index if not exists chat_messages_session_id_created_at_idx
    on public.chat_messages (session_id, created_at);

create index if not exists quizzes_video_id_created_at_idx
    on public.quizzes (video_id, created_at desc);
create index if not exists quizzes_owner_id_created_at_idx
    on public.quizzes (owner_id, created_at desc);
create index if not exists quizzes_session_id_idx
    on public.quizzes (session_id);
-- The unique (quiz_id, question_number) constraint indexes ordered questions.
create index if not exists quiz_attempts_quiz_id_created_at_idx
    on public.quiz_attempts (quiz_id, created_at desc);
create index if not exists quiz_attempts_chat_session_id_idx
    on public.quiz_attempts (chat_session_id);

-- The Flask MVP currently uses a server-side PostgreSQL connection and does not
-- yet have a browser authentication flow.  RLS is enabled now so these tables
-- are not accidentally readable through Supabase's public API.  When Supabase
-- Auth is wired into the UI, add owner-aware policies using auth.uid(); do not
-- make these tables public as a shortcut.
alter table public.videos enable row level security;
alter table public.transcript_chunks enable row level security;
alter table public.processing_jobs enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;
alter table public.quizzes enable row level security;
alter table public.quiz_questions enable row level security;
alter table public.quiz_attempts enable row level security;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

do $$
begin
    if not exists (
        select 1
        from pg_trigger
        where tgname = 'videos_set_updated_at'
          and tgrelid = 'public.videos'::regclass
    ) then
        create trigger videos_set_updated_at
            before update on public.videos
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'transcript_chunks_set_updated_at'
          and tgrelid = 'public.transcript_chunks'::regclass
    ) then
        create trigger transcript_chunks_set_updated_at
            before update on public.transcript_chunks
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'processing_jobs_set_updated_at'
          and tgrelid = 'public.processing_jobs'::regclass
    ) then
        create trigger processing_jobs_set_updated_at
            before update on public.processing_jobs
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'chat_sessions_set_updated_at'
          and tgrelid = 'public.chat_sessions'::regclass
    ) then
        create trigger chat_sessions_set_updated_at
            before update on public.chat_sessions
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'chat_messages_set_updated_at'
          and tgrelid = 'public.chat_messages'::regclass
    ) then
        create trigger chat_messages_set_updated_at
            before update on public.chat_messages
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'quizzes_set_updated_at'
          and tgrelid = 'public.quizzes'::regclass
    ) then
        create trigger quizzes_set_updated_at
            before update on public.quizzes
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'quiz_questions_set_updated_at'
          and tgrelid = 'public.quiz_questions'::regclass
    ) then
        create trigger quiz_questions_set_updated_at
            before update on public.quiz_questions
            for each row execute function public.set_updated_at();
    end if;

    if not exists (
        select 1
        from pg_trigger
        where tgname = 'quiz_attempts_set_updated_at'
          and tgrelid = 'public.quiz_attempts'::regclass
    ) then
        create trigger quiz_attempts_set_updated_at
            before update on public.quiz_attempts
            for each row execute function public.set_updated_at();
    end if;
end;
$$;

create or replace function public.match_transcript_chunks(
    query_embedding vector(384),
    match_threshold real default 0.0,
    match_count integer default 5,
    filter_video_id uuid default null
)
returns table (
    id uuid,
    video_id uuid,
    chunk_index integer,
    content text,
    start_seconds double precision,
    end_seconds double precision,
    metadata jsonb,
    similarity real
)
language sql
stable
set search_path = public, extensions
as $$
    select
        chunks.id,
        chunks.video_id,
        chunks.chunk_index,
        chunks.content,
        chunks.start_seconds,
        chunks.end_seconds,
        chunks.metadata,
        (1 - (chunks.embedding <=> query_embedding))::real as similarity
    from public.transcript_chunks as chunks
    where (filter_video_id is null or chunks.video_id = filter_video_id)
      and (1 - (chunks.embedding <=> query_embedding)) >= match_threshold
    order by chunks.embedding <=> query_embedding
    limit least(greatest(coalesce(match_count, 5), 0), 200);
$$;
