import os
import re

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DATAFRAME_PATH = "dataframe.joblib"
PROCESSED_VIDEOS_PATH = "processed_videos.joblib"
VIDEOS_DIR = "videos"


def get_current_videos():
    if not os.path.isdir(VIDEOS_DIR):
        return []

    return sorted(
        filename
        for filename in os.listdir(VIDEOS_DIR)
        if os.path.isfile(os.path.join(VIDEOS_DIR, filename))
    )


def dataframe_needs_update():
    if not os.path.exists(DATAFRAME_PATH) or not os.path.exists(PROCESSED_VIDEOS_PATH):
        return True

    current_videos = get_current_videos()
    try:
        processed_videos = sorted(joblib.load(PROCESSED_VIDEOS_PATH))
    except Exception:
        return True

    return current_videos != processed_videos


def build_dataframe_if_needed():
    if not dataframe_needs_update():
        return

    current_videos = get_current_videos()
    if not current_videos:
        raise RuntimeError(
            "dataframe.joblib is missing and no videos were found in the videos folder. "
            "Upload or add videos first."
        )

    import process_audio
    import process_data
    import process_json
    import process_video

    os.makedirs("audios", exist_ok=True)
    os.makedirs("json_data", exist_ok=True)
    os.makedirs("clean_json_data", exist_ok=True)

    process_video.to_audio()
    process_audio.to_json()
    process_json.to_clean_json()
    process_data.to_build_dataframe()


def load_dataframe(build_if_needed=False):
    if build_if_needed:
        build_dataframe_if_needed()
    loaded_df = joblib.load(DATAFRAME_PATH)
    if loaded_df.empty or "embedding" not in loaded_df:
        raise RuntimeError("dataframe.joblib does not contain transcript embeddings.")

    return loaded_df, np.vstack(loaded_df.embedding.values)


# Keep web-process import safe when a fresh Render instance has no legacy
# joblib artifacts yet.  The database-backed path can boot independently, and
# the legacy rebuild remains explicit through reload_dataframe().
try:
    df, stored_embeddings = load_dataframe()
except (FileNotFoundError, RuntimeError, ValueError, EOFError):
    df = pd.DataFrame(columns=["video_name", "text", "start", "end", "id", "embedding"])
    stored_embeddings = np.empty((0, 384), dtype=float)
TAG_STOPWORDS = {
    "about", "after", "also", "because", "been", "being", "from", "have",
    "into", "like", "more", "most", "that", "their", "then", "there",
    "these", "they", "this", "when", "where", "which", "with", "your",
    "what", "will", "would", "could", "should", "using", "used", "uses",
    "just", "some", "than", "them", "were", "very", "video", "course"
}

embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return embedding_model


def create_embedding(texts):
    if not texts:
        return []

    return get_embedding_model().encode(texts, convert_to_numpy=True)


def reload_dataframe():
    global df, stored_embeddings
    df, stored_embeddings = load_dataframe(build_if_needed=True)


def get_topic_tags(limit=16):
    tags = []

    for video_name in sorted(df["video_name"].dropna().unique()):
        tags.append(str(video_name))

    text = " ".join(df["text"].dropna().astype(str).tolist()).lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{3,}", text)
    counts = {}
    for word in words:
        if word in TAG_STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1

    for word, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        label = word.replace("-", " ").title()
        if label not in tags:
            tags.append(label)
        if len(tags) >= limit:
            break

    return tags[:limit]

def format_timestamp(seconds):
    seconds = float(seconds)
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def retrieve_chunks(question, top_results=5, video_name=""):
    """
    Returns top matching subtitle chunks.
    """

    if df.empty or stored_embeddings.size == 0:
        raise RuntimeError(
            "No transcript embeddings are available. Upload and process a video first."
        )

    question_embedding = np.array(create_embedding([question])[0])

    search_df = df
    embeddings = stored_embeddings

    if video_name:
        filtered_df = df[df["video_name"] == video_name].copy()
        if not filtered_df.empty:
            search_df = filtered_df.reset_index(drop=True)
            embeddings = np.vstack(search_df.embedding.values)

    similarity = cosine_similarity(
        embeddings,
        question_embedding.reshape(1, -1)
    ).flatten()
    
    max_indices = similarity.argsort()[::-1][:top_results]

    results_df = search_df.iloc[max_indices].copy()
    results_df["start"] = results_df["start"].apply(
        format_timestamp
    )
    results_df["end"] = results_df["end"].apply(
        format_timestamp
    )

    results_df["similarity"] = similarity[max_indices]
    chunks = results_df[
        [
            "video_name",
            "text",
            "start",
            "end",
            "similarity"
        ]
    ].to_dict(orient="records")


    return chunks


def inference(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter API key is missing. Please add OPENROUTER_API_KEY to your .env file."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except Exception as error:
        return f"Could not generate a response: {error}"

    if not response.choices:
        return "No response generated. The selected model may be unavailable. Try changing OPENROUTER_MODEL in .env."

    message = response.choices[0].message.content
    if not message:
        return "The model returned an empty response. Try another OpenRouter model."

    return message


def clean_output(text):
    text = text.replace("**", "")
    text = text.replace("###", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_response(question, chunks):
    """
    Sends context to LLM and returns answer.
    """
    prompt = f"""
    You are an assistant helping students learn from videos.

    Here are subtitle chunks:
    {chunks}

    User question:
    "{question}"

    Instructions:
    - Answer in this exact simple format:

      Direct Answer:
      Give a clear 2-4 sentence answer to the student's question.

      Watch Here:
      Video: video name
      Time: start-end

      Why This Part Helps:
      Explain in 1-2 short sentences why this timestamp is useful.

    - Only use the provided subtitle chunks.
    - Do not quote the chunks unless needed.
    - Do not say "subtitle chunk", "available content", "JSON", or "retrieved chunks".
    - Do not use markdown bold, headings with #, or long bullet lists.
    - If the question is unrelated, say: "This topic is not clearly covered in the course videos."
    """

    response = inference(prompt)

    if response is None:
        response = "No response generated. Please try again."

    return clean_output(response)

def generate_quiz(topic, chunks, quiz_count=5, video_name=""):
    scope = f'Only create questions about the selected video: "{video_name}".' if video_name else "Use any relevant course video."
    prompt = f"""
    You are an AI teaching assistant creating a quiz from course video transcripts.

    Topic:
    "{topic}"

    Quiz size:
    Generate exactly {quiz_count} multiple-choice questions.

    Scope:
    {scope}

    Course chunks:
    {chunks}

    Return two sections:

    STUDENT_QUIZ:
    1. Question text
    A. Option text
    B. Option text
    C. Option text
    D. Option text

    Repeat the same format for every question.
    - Do not show correct answers.
    - Do not show explanations.

    ANSWER_KEY:
    1. Correct: A
    Explanation: short explanation
    Source: video name start-end

    Repeat the same format for every question.

    Use all provided chunks across the quiz. Do not create all questions from only one chunk unless the topic appears only there.
    Each question should be based on a different idea from the retrieved chunks when possible.
    If the topic is not covered in the provided chunks, say: "This topic is not clearly covered in the course videos."
    Do not invent facts outside the provided chunks.
    Do not mention JSON or raw chunks.
    """

    return clean_output(inference(prompt))


def parse_quiz_items(student_quiz, answer_key):
    question_pattern = re.compile(
        r"(?ms)^\s*(\d+)[\).\s]+(.+?)(?=^\s*\d+[\).\s]+|\Z)"
    )
    option_pattern = re.compile(
        r"(?ms)^\s*([A-D])[\).:-]\s*(.+?)(?=^\s*[A-D][\).:-]\s*|\Z)"
    )
    answer_pattern = re.compile(
        r"(?ms)^\s*(\d+)[\).\s]+.*?Correct:\s*([A-D]).*?"
        r"Explanation:\s*(.*?)(?:\n\s*Source:\s*(.*?))?(?=^\s*\d+[\).\s]+|\Z)"
    )

    answers = {}
    for match in answer_pattern.finditer(answer_key):
        number = int(match.group(1))
        answers[number] = {
            "correct": match.group(2).strip(),
            "explanation": clean_output(match.group(3) or ""),
            "source": clean_output(match.group(4) or "")
        }

    items = []
    for match in question_pattern.finditer(student_quiz):
        number = int(match.group(1))
        block = clean_output(match.group(2))
        option_matches = list(option_pattern.finditer(block))
        if len(option_matches) < 4:
            continue

        question_text = block[:option_matches[0].start()].strip()
        options = {
            option_match.group(1): clean_output(option_match.group(2))
            for option_match in option_matches[:4]
        }

        answer = answers.get(number, {})
        items.append({
            "number": number,
            "question": clean_output(question_text),
            "options": options,
            "correct": answer.get("correct", ""),
            "explanation": answer.get("explanation", ""),
            "source": answer.get("source", "")
        })

    return items


def split_quiz_output(quiz_output):
    answer_match = re.search(r"(?im)^\s*-{0,3}\s*ANSWER_KEY\s*:?\s*$", quiz_output)
    if not answer_match:
        return quiz_output.strip(), ""

    student_part = quiz_output[:answer_match.start()]
    answer_part = quiz_output[answer_match.end():]

    student_part = re.sub(r"(?im)^\s*-{0,3}\s*STUDENT_QUIZ\s*:?\s*$", "", student_part, count=1)
    student_part = re.sub(r"(?m)^\s*-{3,}\s*$", "", student_part)
    answer_part = re.sub(r"(?m)^\s*-{3,}\s*$", "", answer_part)

    return student_part.strip(), answer_part.strip()


def repair_quiz_items_from_text(message):
    answer = message.get("answer", "")
    answer_key = message.get("answer_key", "")
    if not answer_key:
        _student_quiz, answer_key = split_quiz_output(answer)

    if not answer_key:
        return message.get("quiz_items", [])

    student_quiz, _unused_answer_key = split_quiz_output(answer)
    parsed_items = parse_quiz_items(student_quiz, answer_key)
    parsed_by_number = {item.get("number"): item for item in parsed_items}
    repaired_items = []

    for item in message.get("quiz_items", []):
        parsed = parsed_by_number.get(item.get("number"), {})
        repaired = dict(item)
        repaired["correct"] = item.get("correct") or parsed.get("correct", "")
        repaired["explanation"] = item.get("explanation") or parsed.get("explanation", "")
        repaired["source"] = item.get("source") or parsed.get("source", "")
        repaired_items.append(repaired)

    return repaired_items or parsed_items


def create_quiz(topic, quiz_count=5, video_name=""):
    if not topic.strip():
        return "Please enter a quiz topic.", "", [], []

    try:
        quiz_count = int(quiz_count)
    except (TypeError, ValueError):
        quiz_count = 5
    quiz_count = max(1, min(quiz_count, 20))

    chunks = retrieve_chunks(topic, top_results=max(10, quiz_count * 2), video_name=video_name)
    quiz_output = generate_quiz(topic, chunks, quiz_count, video_name)
    student_quiz, answer_key = split_quiz_output(quiz_output)
    quiz_items = parse_quiz_items(student_quiz, answer_key)
    return student_quiz, answer_key, chunks, quiz_items


def ask_question(question):

    if not question.strip():
        return "Please enter a question.", []

    chunks = retrieve_chunks(question)

    answer = generate_response(
        question,
        chunks
    )

    return answer, chunks
