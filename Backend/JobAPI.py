import requests
import asyncio
import os
import streamlit as st
import tensorflow_probability as tfp
import tensorflow as tf
from agents import Agent, Runner
from dotenv import load_dotenv
from openai import OpenAI
from ResumeParsing import ResumeParser

load_dotenv()


async def run_agent(job_title: str):
    job_matching_agent = Agent(
        name="Job Matching Assistent",
        instructions=job_title,
        model="gpt-4o-mini",
    )
    result = await Runner.run(
        job_matching_agent, "What kind of jobs do you want to look for"
    )
    result.final_output


def main(file=None):
    if file is None:
        uploaded_file = st.file_uploader(
            "Upload your file", type=["txt", "pdf", "docx"]
        )
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            return uploaded_file
        return None
    else:
        st.success(f"File '{file.name}' passed as parameter!")
        return file


def parse_resume(resume):
    if st.button("Parse File"):
        with st.spinner("Parsing file..."):
            try:
                parsed_data = ResumeParser().parse_resume_file(resume)
            except TypeError:
                st.error("No file provided or file type not supported.")
                return None
            return parsed_data


def get_matching_jobs(parsed_data, job_title):

    parser = ResumeParser()
    skills = parser.match_skills_to_job(parsed_data, job_title)

    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    client = OpenAI(api_key=openai_api_key)
    if not openai_api_key:
        st.error(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your environment."
        )
        return

    # Request more jobs and ensure complete descriptions
    prompt = (
        f"Given the following skills: {', '.join(skills['resume_skills'])}, "
        "Suggest exactly 12-15 jobs that would be a good match based on the provided skills. "
        "For each job, provide ALL of the following fields: Job Title, Location, Monthly Salary, Description (with skill match explanation), Job Description Link, Application Link, and a Match Percentage (showing how much of a match the user is to that job based on their resume, as a percentage from 0% to 100%). "
        "Ensure every job entry is complete and detailed, and do not omit any field. "
        "If the user has few skills, you may suggest fewer jobs, but always provide full details for each job."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful job matching assistant. "
                        "For each job you suggest, provide a longer description explaining why this job matches the user's skills. "
                        "Include a link to a reputable job description page and a link to how to apply for each job. "
                        "Format each job suggestion with: Job Title, Location, Monthly Salary, Description (with skill match explanation), Job Description Link, and Application Link. "
                        "Do not omit any field and ensure every job entry is complete and detailed. "
                        "Ensure that your suggestions are unbiased and do not favor or discriminate against any gender, ethnicity, or background. "
                        "Promote equality and diversity in all job recommendations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2500,
            temperature=0.7,
        )
        job_suggestions = response.choices[0].message.content
        # Suggest possible resume improvements before displaying jobs
        resume_improvement_prompt = (
            f"Given the following parsed resume data: {parsed_data}, and these extracted skills: {skills['resume_skills']}, "
            "analyze the user's unique background, experience, and skill set. "
            "Suggest 3-5 highly personalized, actionable ways the user can improve their resume to stand out for their specific job interests and career goals. "
            "Tailor your advice to their strengths, address any gaps, and recommend enhancements in skill development, resume formatting, and content. "
            "Make sure your suggestions are specific to the user's profile and the job title they are targeting."
        )
        try:
            improvement_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert resume reviewer. "
                            "Provide concise, actionable suggestions to help the user improve their resume for job applications."
                        ),
                    },
                    {"role": "user", "content": resume_improvement_prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            improvements = improvement_response.choices[0].message.content
            st.write("Resume Improvement Suggestions:")
            st.write(improvements)
        except Exception as e:
            st.error(f"Error fetching resume improvement suggestions: {e}")

        st.write("Suggested Jobs:")
        st.write(job_suggestions)
    except Exception as e:
        st.error(f"Error fetching job suggestions from GPT-4: {e}")
    st.write("Extracted Skills:", skills)


if __name__ == "__main__":
    st.title("AI-Powered Job Matching Assistant")
    st.write(
        "Upload your resume and enter a job title to find matching job opportunities."
    )
    resume = main()
    parsed_data = parse_resume(resume)
    title = st.text_input("Enter Job Title")
    if title is None:
        st.error("Please enter a job title")
    else:
        get_matching_jobs(parsed_data, title)
        ResumeParser().generate_formatted_resumes(parsed_data)
