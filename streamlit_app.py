import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Event Feedback Reader", layout="wide")
st.title("Event Feedback Reader")
st.caption("Attendees answer a few questions. An AI reads each answer and tells us how they feel.")


QUESTIONS = [
    ("Programme", "How was the programme?"),
    ("Referral", "Will you refer others to this event? Tell us why."),
    ("Highlights", "What did you enjoy most, or what should we improve?"),
]
MOODS = ["POSITIVE", "NEGATIVE", "UNSURE"]


SAMPLE_ATTENDEES = [
    ["The programme was well organised and inspiring.",
     "Yes, I will definitely invite my friends next time.",
     "I enjoyed the student presentations the most."],
    ["It was good but it started very late.",
     "Yes, I will tell others because the students did well.",
     "The sound system should be improved, I could not hear properly."],
    ["The programme was boring and too long.",
     "No, I would not refer anyone because it was disorganised.",
     "The seats were uncomfortable and the hall was too hot."],
    ["The programme sweet me well well.",
     "Yes, I go tell my people about am.",
     "I like how the young ones presented their work."],
    ["Excellent event, I learned a lot about technology.",
     "Absolutely, everyone should come next year.",
     "The live AI demo was amazing."],
]


threshold = st.sidebar.slider(
    "How sure must the AI be?",
    0.50, 0.99, 0.75, 0.01,
    key="threshold",
    help="Answers where the AI is less sure than this are marked UNSURE.",
)
st.sidebar.caption("Tip: write full sentences. One-word answers like 'Yes' are hard for the AI to read.")

if "responses" not in st.session_state:
    st.session_state["responses"] = []



@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


@st.cache_resource
def load_fill_mask_model():
    return pipeline("fill-mask", model="distilbert-base-uncased")


def read_answers(texts):
    """Ask the AI for the mood of each answer. Returns one result per answer."""
    results = load_sentiment_model()(texts, truncation=True)
    readings = []
    for r in results:
        mood = r["label"] if r["score"] >= threshold else "UNSURE"
        signed = r["score"] if r["label"] == "POSITIVE" else -r["score"]
        readings.append({"mood": mood, "score": round(r["score"], 3), "signed": signed})
    return readings


def overall_mood(readings):
    """One mood for the whole person: the average of their answers."""
    average = sum(r["signed"] for r in readings) / len(readings)
    if average >= 0.3:
        return "POSITIVE"
    if average <= -0.3:
        return "NEGATIVE"
    return "MIXED"


def save_person(answers_by_question, readings):
    """Store one attendee's answers in the list of responses."""
    ids = [row["respondent"] for row in st.session_state["responses"]]
    person_id = max(ids, default=0) + 1
    for (short, question, answer), reading in zip(answers_by_question, readings):
        st.session_state["responses"].append(
            {
                "respondent": person_id,
                "topic": short,
                "question": question,
                "answer": answer,
                "mood": reading["mood"],
                "score": reading["score"],
            }
        )


def show_mood(mood, message):
    if mood == "POSITIVE":
        st.success(message)
    elif mood == "NEGATIVE":
        st.error(message)
    else:
        st.info(message)


tab1, tab2, tab3, tab4 = st.tabs(
    ["Event survey", "Results so far", "One sentence", "Missing word"]
)


with tab1:
    st.subheader("Tell us about the event")
    st.write("Answer in a sentence or two. You can skip a question.")

    with st.form("survey_form", clear_on_submit=True):
        typed = {}
        for short, question in QUESTIONS:
            typed[short] = st.text_area(
                question,
                key=f"answer_{short}",
                height=90,
                placeholder="Write your answer here...",
            )
        submitted = st.form_submit_button("Submit my answers")

    if submitted:
        filled = [
            (short, question, typed[short].strip())
            for short, question in QUESTIONS
            if typed[short].strip()
        ]
        if not filled:
            st.warning("Please answer at least one question.")
        else:
            with st.spinner("The AI is reading your answers..."):
                readings = read_answers([answer for _, _, answer in filled])
            save_person(filled, readings)
            st.session_state["last_person"] = (filled, readings)

    if "last_person" in st.session_state:
        filled, readings = st.session_state["last_person"]
        st.subheader("Thank you! Here is what the AI understood")
        for (short, question, answer), reading in zip(filled, readings):
            st.write(f"**{question}**")
            st.write(f"\"{answer}\"")
            show_mood(reading["mood"], f"{reading['mood']} (score: {reading['score']})")
        verdict = overall_mood(readings)
        show_mood(verdict, f"Overall, you sound: {verdict}")


with tab2:
    st.subheader("What everyone said")

    left, right = st.columns(2)
    if left.button("Load 5 practice attendees", key="sample_btn"):
        for answers in SAMPLE_ATTENDEES:
            rows = [(s, q, a) for (s, q), a in zip(QUESTIONS, answers)]
            save_person(rows, read_answers([a for _, _, a in rows]))
    if right.button("Clear all responses", key="clear_btn"):
        st.session_state["responses"] = []
        st.session_state.pop("last_person", None)

    df = pd.DataFrame(st.session_state["responses"])

    if df.empty:
        st.info("No answers yet. Fill in the survey, or load the practice attendees.")
    else:
        total_answers = len(df)
        counts = df["mood"].value_counts().reindex(MOODS, fill_value=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attendees", df["respondent"].nunique())
        c2.metric("Positive answers", f"{counts['POSITIVE'] / total_answers:.0%}")
        c3.metric("Negative answers", f"{counts['NEGATIVE'] / total_answers:.0%}")
        c4.metric("Unsure", f"{counts['UNSURE'] / total_answers:.0%}")

        referral = df[df["topic"] == "Referral"]
        if not referral.empty:
            share = (referral["mood"] == "POSITIVE").mean()
            st.metric("Positive about referring others to the event", f"{share:.0%}")

        st.write("**Mood for each question**")
        by_question = pd.crosstab(df["topic"], df["mood"]).reindex(columns=MOODS, fill_value=0)
        st.bar_chart(by_question)

        st.write("**All answers**")
        st.dataframe(df[["respondent", "question", "answer", "mood", "score"]])

        st.download_button(
            "Download results (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="event_feedback_results.csv",
            mime="text/csv",
        )
        st.caption(
            "The AI reads English best. Pidgin or mixed-language answers may be "
            "marked UNSURE or misread."
        )


with tab3:
    st.write("Type a sentence. The model will say if it is POSITIVE or NEGATIVE.")
    text = st.text_area(
        "Your sentence", "I love learning about artificial intelligence.", key="single_text"
    )

    if st.button("Check sentiment", key="single_btn"):
        if not text.strip():
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("Thinking..."):
                result = load_sentiment_model()(text, truncation=True)[0]
            message = f"{result['label']} (score: {round(result['score'], 3)})"
            if result["label"] == "POSITIVE":
                st.success(message)
            else:
                st.error(message)


MASK = "[MASK]"
with tab4:
    st.write(f"Type a sentence and use {MASK} where a word is missing.")
    sentence = st.text_area("Your sentence", f"We are about {MASK} go home.", key="mask_text")

    if st.button("Guess the word", key="mask_btn"):
        if sentence.count(MASK) != 1:
            st.warning(f"Please use {MASK} exactly one time in your sentence.")
        else:
            with st.spinner("Thinking..."):
                guesses = load_fill_mask_model()(sentence, top_k=5)
            st.write("Top 5 guesses:")
            for guess in guesses:
                st.write(f"**{guess['token_str']}**  (score: {round(guess['score'], 3)})")
                st.progress(min(float(guess["score"]), 1.0))
                