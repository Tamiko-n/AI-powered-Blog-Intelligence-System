import pandas as pd
import time
import requests
import ast
from urllib.parse import urlparse
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from newspaper import Article
from nlp_engine import NLPEngine
from visuals import create_analysis_dashboard

SERP_API_KEY = "959dbda43cf7921c14c68d0ff5a85e53a4272a90da7de2af7808961664823b55"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

nlp = NLPEngine()


def get_links(keyword, page):
    query = f'{keyword} blog ("comments" OR "discussion") (site:wordpress.com OR site:blogspot.com OR site:dev.to)'
    params = {"q": query, "api_key": SERP_API_KEY, "start": page * 10}

    try:
        search = GoogleSearch(params)
        return [r.get("link") for r in search.get_dict().get("organic_results", []) if r.get("link")]
    except:
        return []


def process_blog(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=12)
        if response.status_code != 200:
            return f"Error {response.status_code}"

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        article = Article(url)
        article.download(input_html=html)
        article.parse()

        if not article.text or len(article.text) < 200:
            return "Insufficient content"

        author_name = "Unknown"
        if article.authors:
            author_name = article.authors[0]

        if "view my complete profile" in author_name.lower() or author_name == "Unknown":
            try:
                profile_link = soup.select_one('a.profile-link, .g-profile')
                if profile_link and 'href' in profile_link.attrs:
                    p_url = profile_link['href']
                    p_res = requests.get(p_url, headers=HEADERS, timeout=5)
                    p_soup = BeautifulSoup(p_res.text, 'html.parser')
                    p_title = p_soup.find('title').get_text()
                    author_name = p_title.replace("User Profile:", "").strip()
            except:
                pass

        comment_tags = soup.select('.comment-content, .comment-body, .comment-text, article.comment, .blockquote')
        comments = [t.get_text(strip=True) for t in comment_tags if len(t.get_text()) > 15]

        sentiment = nlp.get_comment_sentiment(comments)

        return {
            "url": url,
            "title": article.title,
            "author": author_name,
            "date": article.publish_date if article.publish_date else "N/A",
            "text": article.text,
            "comment_count": len(comments),
            "sentiment": sentiment,
            "comments": comments
        }

    except Exception as e:
        return f"System Fail: {str(e)}"


def run_data_mining(keyword,callback=None):
    results = []
    domains = set()
    blogs_with_comments_count = 0
    page = 0

    print(f"\n🚀 Pipeline started. Target: 10 blogs (Min 5 with comments) from ≥3 domains.")

    while (len(results) < 10 or len(domains) < 3) and page < 10:
        links = get_links(keyword, page)

        if not links:
            print("❌ No more links found.")
            break

        for link in links:
            if len(results) >= 10 and len(domains) >= 3 and blogs_with_comments_count >= 5:
                break

            if link in [r['url'] for r in results]:
                continue

            domain = urlparse(link).netloc
            print(f"🧐 Analyzing: {domain}...", end=" ", flush=True)

            data = process_blog(link)

            if isinstance(data, dict):
                has_comments = data['comment_count'] > 0

                if has_comments:
                    results.append(data)
                    domains.add(domain)
                    blogs_with_comments_count += 1
                    print(f"✅ SUCCESS ({len(results)}/10, {blogs_with_comments_count} with comments)")

                elif not has_comments and blogs_with_comments_count >= 5 and len(results) < 10:
                    results.append(data)
                    domains.add(domain)
                    print(f"✅ SUCCESS ({len(results)}/10, No comments)")

                else:
                    print("⏭️ Skipping: Need at least 5 blogs with comments first.")
            else:
                print(f"❌ {data}")

        page += 1
        time.sleep(1)

    if results:
        df = pd.DataFrame(results)

        print("\n🤖 Step 1: Running Local AI (Phi-3 via Ollama)...")

        summaries, motives, tones = [], [], []

        for idx, row in df.iterrows():
            print(f"   ✨ Processing blog {idx+1}/{len(results)}...", end="\r")

            summary, motive, tone = nlp.analyze_content_local(row['text'])

            summaries.append(summary)
            motives.append(motive)
            tones.append(tone)

            time.sleep(1)

        df['summary'] = summaries
        df['motive'] = motives
        df['tone'] = tones

        print("\n🤖 Step 2: Running Hybrid LDA (Topic Clustering)...")
        all_text = df['text'].tolist()
        topic_results = nlp.run_hybrid_lda(all_text)

        df['topic_theme'] = [
            topic_results['labels'][topic_id]
            for topic_id in topic_results['assignments']
        ]

        print("\n🔑 Step 3: Extracting TF-IDF Keywords...")
        all_keywords = nlp.extract_corpus_keywords(df['text'].tolist())
        df['unique_keywords'] = [", ".join(k) for k in all_keywords]

        df.to_csv("analyzed_blogs.csv", index=False)
        print("💾 Saved to analyzed_blogs.csv")

        print("\n" + "=" * 60)
        print("🔥 FINAL DATA MINING REPORT")
        print("=" * 60)

        for i, row in df.iterrows():
            comment_status = "💬 ENGAGED" if row['comment_count'] > 0 else "⚪ NO COMMENTS"

            print(f"\n[{i+1}] {row['title']} ({comment_status})")
            print(f"🔗 {row['url']}")
            print(f"👤 Author: {row['author']} | 📅 Date: {row['date']}")
            print(f"🎯 Topic Theme: {row['topic_theme']}")
            print(f"📝 SUMMARY: {row['summary']}")
            print(f"🎯 Motive: {row['motive']} | 🎭 Tone: {row['tone']}")
            print(f"💬 Comments: {row['comment_count']} | Sentiment: {row['sentiment']:.2f}")
            print(f"🔑 Keywords: {row['unique_keywords']}")
            print("-" * 40)


    else:
        print("\n⚠️ Criteria not met. No blogs collected.")


if __name__ == "__main__":
    run_data_mining()