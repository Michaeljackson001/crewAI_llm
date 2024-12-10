import streamlit as st
from dotenv import load_dotenv
from crewai import Crew
from tasks import MeetingPrepTasks
from agents import MeetingPrepAgents
from langchain_groq import ChatGroq
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = "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"  # Replace with actual API key

st.set_page_config(page_title="Meeting Preparation Crew", layout="wide", page_icon="✨")


st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .stApp {
            background-color: #000000;
        }
        .stTextInput > label, .stTextArea > label, .stButton button {
            color: white;
            font-weight: bold;
        }
        .stButton button {
            background-color: #333333;
            border: none;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #555555;
        }
    </style>
    """,
    unsafe_allow_html=True
)

llm = ChatGroq(
    temperature=0.9,
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192",
)

st.title("✨ Meeting Preparation Crew")
st.caption("**This tool helps you prepare effectively for your meetings.**")

st.sidebar.header("Configure Your Meeting")
participant_emails = st.sidebar.text_input(
    "Enter the emails of the meeting participants (comma-separated):"
)
meeting_context = st.sidebar.text_area("Enter the context of the meeting:")
meeting_objective = st.sidebar.text_area("Enter your objective for this meeting:")

# Main section
st.subheader("Meeting Details Preview")
if not participant_emails or not meeting_context or not meeting_objective:
    st.warning("Please complete all fields in the sidebar to proceed.")
else:
    st.write("### Meeting Participants")
    st.write(participant_emails)
    st.write("### Meeting Context")
    st.write(meeting_context)
    st.write("### Meeting Objectives")
    st.write(meeting_objective)

# Process button
if st.button("Run Meeting Prep Crew"):
    st.info("Processing... Please wait.")

    try:
        # Initialize tasks and agents
        tasks = MeetingPrepTasks()
        agents = MeetingPrepAgents()

        # Create agents
        research_agent = agents.research_agent(llm)
        industry_analysis_agent = agents.industry_analysis_agent(llm)
        meeting_strategy_agent = agents.meeting_strategy_agent(llm)
        summary_and_briefing_agent = agents.summary_and_briefing_agent(llm)

        # Create tasks
        research_task = tasks.research_task(research_agent, participant_emails, meeting_context)
        industry_analysis_task = tasks.industry_analysis_task(industry_analysis_agent, participant_emails, meeting_context)
        meeting_strategy_task = tasks.meeting_strategy_task(meeting_strategy_agent, meeting_context, meeting_objective)
        summary_and_briefing_task = tasks.summary_and_briefing_task(summary_and_briefing_agent, meeting_context, meeting_objective)

        # Link tasks
        meeting_strategy_task.context = [research_task, industry_analysis_task]
        summary_and_briefing_task.context = [research_task, industry_analysis_task, meeting_strategy_task]

        # Initialize Crew
        crew = Crew(
            agents=[
                research_agent,
                industry_analysis_agent,
                meeting_strategy_agent,
                summary_and_briefing_agent,
            ],
            tasks=[
                research_task,
                industry_analysis_task,
                meeting_strategy_task,
                summary_and_briefing_task,
            ],
        )

        # Run Crew
        result = crew.kickoff()

        # Display results
        if isinstance(result, str):
            st.success("Meeting Preparation Output")
            st.text_area("Results", result, height=300)
        elif isinstance(result, dict):
            st.success("Meeting Preparation Output")
            st.json(result)
        else:
            st.warning("Unexpected result format. Please check the CrewAI output.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
