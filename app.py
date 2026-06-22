import streamlit as st
import pandas as pd
import time
from visuals import create_analysis_dashboard
from file1 import run_data_mining
from comment_generation import generate_human_like_comment

st.set_page_config(page_title="AI Blog Intelligence System", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"

if "keyword" not in st.session_state:
    st.session_state.keyword = ""

if "df" not in st.session_state:
    st.session_state.df = None

if st.session_state.page == "home":

    st.title("🧠 AI Blog Intelligence System")

    st.write("""
    This system scrapes blogs, analyzes sentiment, extracts topics,
    and generates AI-powered insights using local NLP models.
    """)

    st.info("🔍 Features: Scraping • Sentiment Analysis • LDA Topics • AI Summaries • Visual Dashboard")

    if st.button("🚀 Click Here to Continue"):
        st.session_state.page = "input"
        st.rerun()

elif st.session_state.page == "input":

    st.title("🔎 Enter Keyword for Analysis")

    keyword = st.text_input("Type your keyword (e.g. AI, Python, Blockchain)")

    if st.button("Start Analysis"):

        if keyword.strip() == "":
            st.warning("Please enter a keyword first.")
        else:
            st.session_state.keyword = keyword
            st.session_state.page = "processing"
            st.rerun()

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "processing":

    st.title("⚙️ Live AI Processing Pipeline")

    keyword = st.session_state.keyword

    st.write(f"🔍 Searching blogs for: **{keyword}**")

    log_box = st.empty()
    progress = st.progress(0)

    with st.status("🚀 Running pipeline...", expanded=True) as status:

        def progress_callback(step, message, percent):
            log_box.write(message)
            progress.progress(percent)

        results = run_data_mining(keyword, progress_callback)

        status.update(label="✅ Data Collection Complete", state="complete")

    try:
        df = pd.read_csv("analyzed_blogs.csv")
        st.session_state.df = df

    except:
        st.error("❌ No data generated. Try another keyword.")
        st.stop()

    st.success("✅ Analysis Complete!")

    st.subheader("📄 Analyzed Blogs")

    for i, row in df.iterrows():

        with st.expander(f"📌 {row['title']}"):

            st.write("🔗 URL:", row["url"])
            st.write("👤 Author:", row["author"])
            st.write("📅 Date:", row["date"])
            st.write("🧠 Summary:", row["summary"])
            st.write("🎯 Motive:", row["motive"])
            st.write("🎭 Tone:", row["tone"])
            st.write("💬 Comments:", row["comment_count"])
            st.write("😊 Sentiment:", row["sentiment"])
            st.write("🔑 Keywords:", row["unique_keywords"])
            st.write("🏷 Topic:", row["topic_theme"])

    st.subheader("📊 Visual Analytics")

    create_analysis_dashboard(df.to_dict("records"))

    if st.button("💬 Generate AI Comment"):
        with st.spinner("Generating human-like comment..."):
            comment, meta = generate_human_like_comment()

            if comment:
                st.success("Generated Comment:")
                st.write(comment)

                st.info(f"Based on blog: {meta}")
            else:
                st.error(meta)

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()
