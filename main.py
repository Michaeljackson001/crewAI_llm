from dotenv import load_dotenv
from crewai import Crew
from tasks import MeetingPrepTasks
from agents import MeetingPrepAgents
from langchain_groq import ChatGroq

import os
os.environ["GROQ_API_KEY"]="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"
llm=ChatGroq(
    temperature=0.9,
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)


def main():
    load_dotenv()
    
    print("## Welcome to the Meeting Prep Crew")
    print('-------------------------------')
    meeting_participants = input("What are the emails for the participants (other than you) in the meeting?\n")
    meeting_context = input("What is the context of the meeting?\n")
    meeting_objective = input("What is your objective for this meeting?\n")

    tasks = MeetingPrepTasks()
    agents = MeetingPrepAgents()
    
    # create agents
    research_agent = agents.research_agent(llm)
    industry_analysis_agent = agents.industry_analysis_agent(llm)
    meeting_strategy_agent = agents.meeting_strategy_agent(llm)
    summary_and_briefing_agent = agents.summary_and_briefing_agent(llm)
    
    # create tasks
    research_task = tasks.research_task(research_agent, meeting_participants, meeting_context)
    industry_analysis_task = tasks.industry_analysis_task(industry_analysis_agent, meeting_participants, meeting_context)
    meeting_strategy_task = tasks.meeting_strategy_task(meeting_strategy_agent, meeting_context, meeting_objective)
    summary_and_briefing_task = tasks.summary_and_briefing_task(summary_and_briefing_agent, meeting_context, meeting_objective)
    
    meeting_strategy_task.context = [research_task, industry_analysis_task]
    summary_and_briefing_task.context = [research_task, industry_analysis_task, meeting_strategy_task]
    
    crew = Crew(
      agents=[
        research_agent,
        industry_analysis_agent,
        meeting_strategy_agent,
        summary_and_briefing_agent
      ],
      tasks=[
        research_task,
        industry_analysis_task,
        meeting_strategy_task,
        summary_and_briefing_task
      ]
    )
    
    result = crew.kickoff()
    
    print(result)
    
if __name__ == "__main__":
    main()