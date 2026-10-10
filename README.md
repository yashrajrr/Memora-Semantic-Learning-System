# Memora: Semantic Learning System

Memora is a semantic learning assistant for educational videos. It converts uploaded lecture videos into searchable transcript chunks, creates local embeddings for semantic retrieval, and lets students ask questions or generate interactive quizzes from the video content.

The project uses a Retrieval-Augmented Generation (RAG) workflow: retrieve the most relevant video transcript chunks first, then send only that focused context to an LLM for answers or quiz generation.

## Features

- Upload and process educational videos from the web interface.
- Convert video to audio with ffmpeg.
- Transcribe audio locally with Whisper.
- Clean transcript segments into timestamped chunks.
- Generate local semantic embeddings with Sentence Transformers.
- Ask questions against uploaded video content.
- Generate interactive quizzes from selected tags, custom topics, specific videos, or all videos.
- Submit quiz answers in-page without refreshing the whole screen.
- View quiz reports with history, scores, strong topics, and weak topics.
- Save chat and quiz sessions in `chat_history.json`.
- Cache processed transcript embeddings in `dataframe.joblib`.

## How It Works

Memora has two main phases: processing and retrieval.

### 1. Video Processing

When a video is uploaded:

1. The video is saved in `videos/`.
2. `process_video.py` extracts audio into `audios/`.
3. `process_audio.py` transcribes the audio using Whisper and saves raw JSON in `json_data/`.
4. `process_json.py` cleans transcript segments and saves structured chunks in `clean_json_data/`.
5. `process_data.py` embeds each chunk and saves everything into `dataframe.joblib`.

Each clean chunk contains:

- video name
- transcript text
- start timestamp
- end timestamp
- embedding vector

### 2. Question Answering and Quiz Generation

When the user asks a question or requests a quiz:

1. The user query/topic is embedded locally.
2. Memora compares that query embedding with saved transcript embeddings using cosine similarity.
3. The top matching chunks are selected.
4. Only those chunks are sent to the configured LLM through OpenRouter.
5. The LLM generates an answer or quiz from the retrieved video context.

## Why It Feels Fast

Memora is fast during chat and quiz usage because the expensive embedding work is already done during video processing.

It does not re-embed all videos every time you ask something. Instead:

1. Transcript chunk embeddings are created once and saved in `dataframe.joblib`.
2. During chat, only the current question/topic is embedded.
3. That one query vector is compared against the saved vectors using fast local vector math.

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is a small, efficient Sentence Transformers model. It is downloaded/cached from Hugging Face the first time, then it runs locally on your machine. Embeddings are not generated through a Hugging Face API.

The live retrieval step is fast because it mainly does:

```python
cosine_similarity(stored_embeddings, question_embedding)
```

That is much cheaper than transcribing videos or calling an LLM over the full transcript.

## Local vs API Components

Memora uses a mix of local models and API-based LLM generation:

| Task | Tool | Runs where |
| --- | --- | --- |
| Video to audio | ffmpeg | Local |
| Speech transcription | Whisper | Local |
| Embeddings | Sentence Transformers | Local |
| Similarity search | scikit-learn cosine similarity | Local |
| Final answer / quiz generation | OpenRouter model | API |

## Project Structure

```text
.
├── app.py                  # Flask web application
├── main.py                 # Command-line workflow
├── get_output.py           # Retrieval, LLM prompts, quiz parsing
├── chat_history.py         # Session and quiz history storage
├── process_video.py        # Video to audio conversion
├── process_audio.py        # Whisper transcription
├── process_json.py         # Transcript cleaning
├── process_data.py         # Embedding generation
├── templates/
│   ├── index.html          # Chat and quiz UI
│   ├── videos.html         # Video upload page
│   └── quiz_report.html    # Quiz report page
├── videos/                 # Uploaded videos
├── audios/                 # Extracted audio files
├── json_data/              # Raw Whisper output
├── clean_json_data/        # Clean timestamped transcript chunks
├── dataframe.joblib        # Saved chunks + embeddings
├── processed_videos.joblib # Processed video cache
├── chat_history.json       # Chat and quiz history
├── requirements.txt
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose app runner
├── .dockerignore           # Files excluded from Docker build context
├── .env.example            # Example environment variables
├── .env                    # Local environment variables, not committed
└── LICENSE
```

## Requirements

- Python 3.10 or newer recommended for local runs
- Docker Desktop for Docker runs
- System ffmpeg for local runs
- Internet connection for first-time model downloads and OpenRouter calls
- OpenRouter API key

System ffmpeg is required for video-to-audio conversion when running locally. The Docker setup installs ffmpeg automatically.

## Installation

1. Clone or download the project.

2. Open a terminal in the project folder.

3. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Command Prompt:

```bash
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=your_model_id_here
WHISPER_MODEL=tiny
```

Example:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1:free
WHISPER_MODEL=tiny
```

Notes:

- `OPENROUTER_API_KEY` is required for LLM answers and quiz generation.
- `OPENROUTER_MODEL` controls which OpenRouter model is used.
- `WHISPER_MODEL` controls local transcription speed and quality.
- Do not commit your `.env` file.

## Running the Web App

Start the Flask app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Health checks are available at `http://127.0.0.1:5000/healthz`. In production,
set `REQUIRE_DATABASE=true` so the endpoint reports `503` when PostgreSQL is
unavailable.

## Supabase and Render deployment

The repository now contains the version-controlled Supabase migration at
`supabase/migrations/20261010000000_initial_schema.sql`. It creates the
pgvector-backed schema, processing-job tables, chat/quiz tables, indexes, and
the `match_transcript_chunks` function. The application keeps schema changes
explicit; it does not run migrations automatically on every web-service boot.

Install the Supabase CLI, then authenticate and link the project:

```bash
npx supabase login
npx supabase init
npx supabase link --project-ref yrlnigaomvcklhohfdcc
```

Apply the reviewed migration with the linked CLI:

```bash
npx supabase db push
```

Alternatively, from a trusted machine or a Render one-off shell with
`DATABASE_URL` set:

```bash
python migrate.py
```

Set `DATABASE_URL` to the server-side Supabase PostgreSQL connection string.
Do not put a database password, service-role key, or other privileged value in
templates, browser JavaScript, Git, or `.env.example`. The current MVP has no
browser authentication yet, so the schema enables RLS without public policies;
the server must use its private PostgreSQL connection. Owner-aware policies
should be added when Supabase Auth is introduced.

Render can deploy the web service from `render.yaml`. It uses Gunicorn, binds
to Render's `$PORT`, installs FFmpeg through the Dockerfile when Docker is
selected, and probes `/healthz`. Configure at least `FLASK_SECRET_KEY`,
`DATABASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `WHISPER_MODEL`, and
`REQUIRE_DATABASE=true` in the Render dashboard. The current upload route is
still synchronous and local-file based; use persistent Supabase Storage and a
database-backed worker before accepting long-running production uploads.

Main pages:

- `/` - chat and quiz sessions
- `/videos` - upload and process videos
- `/quiz-report` - quiz history, scores, weak topics, strong topics


## Running With Docker

Docker is the easiest way to share and run the project because it installs Python dependencies and ffmpeg inside the container.

1. Make sure Docker Desktop is installed and running.

2. Create your local environment file:

Windows PowerShell:

```powershell
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

3. Edit `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

4. Build and start the app:

```bash
docker compose up --build
```

5. Open the app:

```text
http://127.0.0.1:5000/
```

After the first build, you can usually start it with:

```bash
docker compose up
```

Use `docker compose up --build` again after changing `Dockerfile`, `requirements.txt`, or dependencies.

To stop the app:

```bash
docker compose down
```

Docker keeps generated project data in the mounted folders and keeps downloaded model cache in a Docker volume.

## Running the CLI

You can also use the command-line workflow:

```bash
python main.py
```

The CLI checks whether videos have changed, processes them if needed, then lets you ask questions or generate quizzes.

## Uploading Videos

1. Go to `/videos`.
2. Select a video file.
3. Click **Upload and Process**.
4. Wait for the processing message.

Supported extensions:

```text
mp4, webm, mov, mkv, avi
```

Processing may take time depending on:

- video length
- CPU speed
- Whisper model size
- whether models are already downloaded

## Whisper Model Speed vs Quality

`WHISPER_MODEL=tiny` is the default because it is much faster for uploads.

Common options:

```env
WHISPER_MODEL=tiny
WHISPER_MODEL=base
WHISPER_MODEL=small
WHISPER_MODEL=medium
```

Tradeoff:

- `tiny` is fastest, lower accuracy.
- `base` is still fast, better accuracy.
- `small` is slower, better transcription.
- `medium` is much slower on CPU.

If upload processing feels too slow, use:

```env
WHISPER_MODEL=tiny
```

If transcription quality is weak, try:

```env
WHISPER_MODEL=base
```

or:

```env
WHISPER_MODEL=small
```

## Changing the LLM Model

The LLM model is selected in `.env`:

```env
OPENROUTER_MODEL=model/provider-id
```

For example:

```env
OPENROUTER_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1:free
```

or:

```env
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

Use a model available in your OpenRouter account. Free models may be rate-limited, so if you see a `429` error, wait and retry or switch to another model.

## Changing the Embedding Model

The embedding model is currently set in `get_output.py` and `process_data.py`:

```python
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

To use another local Sentence Transformers model:

1. Replace the model name in both files.
2. Delete or rebuild `dataframe.joblib`.
3. Re-run video processing so all chunks are embedded with the new model.

Example alternatives:

```text
sentence-transformers/all-mpnet-base-v2
BAAI/bge-small-en-v1.5
BAAI/bge-base-en-v1.5
intfloat/e5-small-v2
```

Important: the query embedding model and stored chunk embedding model must match. If you change the model only in one place, retrieval quality may break because vectors will not be compatible.

## Rebuilding Embeddings

If `dataframe.joblib` or `processed_videos.joblib` is missing, or if the files in `videos/` changed, the app automatically processes the videos and rebuilds `dataframe.joblib` before loading retrieval data.

You can also manually rebuild embeddings with:

```bash
python process_data.py
```

## Data Files

Generated files:

- `audios/*.mp3`
- `json_data/*.json`
- `clean_json_data/*.json`
- `dataframe.joblib`
- `processed_videos.joblib`
- `chat_history.json`

These are runtime/project data files and may become large. Decide whether to commit them based on your project needs.


## Sharing on GitHub

Commit the source code and configuration files, but do not commit secrets or large generated artifacts.

Recommended files to commit:

- Python source files such as `app.py`, `get_output.py`, and `process_*.py`
- `templates/`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`
- `README.md`
- `LICENSE`

Do not commit:

- `.env`
- `.venv/`
- `__pycache__/`
- large model files
- generated `.joblib` files, unless you intentionally want to ship prebuilt embeddings

Your current `.gitignore` already ignores `.env`, virtual environments, Python cache folders, and `.joblib` files.

A new user can run the project with:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
cp .env.example .env
docker compose up --build
```

On Windows PowerShell, use:

```powershell
copy .env.example .env
docker compose up --build
```

If you do not commit videos or `dataframe.joblib`, users should upload videos through `/videos`. The app will process them and create `dataframe.joblib` automatically.

## Troubleshooting

### Upload looks instant but video is not searchable

Check `/videos` for a flash message. Also verify that files were created in:

```text
audios/
json_data/
clean_json_data/
```

If no transcript files appear, processing did not complete.

### ffmpeg not found

Memora requires system ffmpeg when running locally. Install ffmpeg and make sure it is available on PATH.

With Docker, ffmpeg is installed automatically by the Dockerfile.

### LLM returns 429

This means the selected OpenRouter model is rate-limited. Try:

- waiting a few minutes
- switching `OPENROUTER_MODEL`
- using a paid model
- adding your own provider key in OpenRouter

### Quiz score is wrong

Memora parses `ANSWER_KEY` from the LLM output and stores correct answers in `chat_history.json`. If the LLM returns a malformed quiz, scoring can be affected. The app includes repair logic for common formats such as:

```text
ANSWER_KEY
ANSWER_KEY:
STUDENT_QUIZ
STUDENT_QUIZ:
```

### Retrieval gives weak answers

Possible reasons:

- transcript quality is poor
- video has unclear audio
- `WHISPER_MODEL=tiny` missed words
- the question is unrelated to uploaded videos
- the embedding model is too small for your domain

Try a better Whisper model or a stronger embedding model.

## License

Memora is free to use under the MIT License. See [LICENSE](LICENSE).

