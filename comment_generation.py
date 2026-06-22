import pandas as pd
import ast
import ollama
import re


def generate_human_like_comment():
    try:
        df = pd.read_csv("analyzed_blogs.csv")
    except FileNotFoundError:
        return None, "❌ analyzed_blogs.csv not found"

    if df.empty:
        return None, "⚠️ CSV is empty"

    blog_to_analyze = df.sort_values(by='comment_count', ascending=False).iloc[0]

    summary = blog_to_analyze['summary']
    raw_comments = blog_to_analyze['comments']

    try:
        existing_discussion = ast.literal_eval(raw_comments) if pd.notna(raw_comments) else []
    except:
        existing_discussion = []

    discussion_snippet = "\n".join([str(c)[:200] for c in existing_discussion[:5]])

    prompt = f"""
    Write a short casual comment for this blog.

    SUMMARY: {summary}
    PREVIOUS COMMENTS: {discussion_snippet if discussion_snippet else "None"}

    Rules:
    - max 40 words
    - use I or me
    - very simple English (like teenager writing)
    - no labels or formatting
    """

    try:
        response = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}]
        )

        comment = response['message']['content'].strip()

        comment = re.sub(r'^(Comment|Response):\s*', '', comment, flags=re.IGNORECASE)
        comment = comment.replace('"', '')

        output_file = "generated_comment.csv"

        pd.DataFrame([{
            "title": blog_to_analyze['title'],
            "summary": summary,
            "comment": comment,
            "url": blog_to_analyze['url'],
            "source": "dev.to"
        }]).to_csv(output_file, index=False)

        return comment, blog_to_analyze['title']

    except Exception as e:
        return None, f"❌ Mistral error: {str(e)[:100]}"