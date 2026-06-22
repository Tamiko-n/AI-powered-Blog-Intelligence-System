import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import streamlit as st


def create_analysis_dashboard(results):
    if not results:
        st.warning("No data to visualize.")
        return

    df = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    plt.subplots_adjust(hspace=0.4, right=0.80)

    sns.barplot(
        ax=axes[0, 0],
        x=df.index + 1,
        y='sentiment',
        data=df,
        hue=df.index + 1,
        palette="RdYlGn",
        legend=False
    )

    axes[0, 0].set_title("Sentiment Analysis by Blog Index")
    axes[0, 0].set_xlabel("Blog ID")
    axes[0, 0].set_ylabel("Sentiment Score (-1 to 1)")
    axes[0, 0].set_ylim(-1, 1)
    axes[0, 0].axhline(0, color='black', lw=1)

    all_motives = []
    for m in df['motive']:
        all_motives.extend([x.strip() for x in m.split(',')])

    motive_counts = pd.Series(all_motives).value_counts()

    axes[0, 1].pie(
        motive_counts,
        labels=motive_counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette("pastel")
    )
    axes[0, 1].set_title("Content Motive Distribution")

    sns.scatterplot(
        ax=axes[1, 0],
        x='comment_count',
        y='sentiment',
        size='comment_count',
        hue='tone',
        style='tone',
        data=df,
        s=100,
        legend='brief'
    )

    axes[1, 0].set_title("Engagement vs Sentiment (Tone-based)")
    axes[1, 0].set_xlabel("Number of Comments")

    axes[1, 0].legend(
        bbox_to_anchor=(1.25, 1),
        loc='upper left',
        borderaxespad=0
    )

    all_comments = " ".join([" ".join(c) for c in df['comments']])

    axes[1, 1].set_title("Discussion Word Cloud")
    axes[1, 1].axis('off')

    if all_comments.strip():
        wordcloud = WordCloud(
            width=500,
            height=300,
            background_color='white',
            colormap='viridis'
        ).generate(all_comments)

        axes[1, 1].imshow(wordcloud, interpolation='bilinear')

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    st.pyplot(fig)

    fig.savefig('data_mining_dashboard.png', dpi=300, bbox_inches='tight')