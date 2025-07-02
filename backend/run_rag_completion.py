from dotenv import load_dotenv
import os

from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex

load_dotenv

def run_rag_completion(
        documents,
        query_text: str,
        job_title: str,
        job_description: str,
        embedding_model: str = 'BAAI/bge-en-icl',
        generative_model: str = "Qwen/Qwen3-235B-A22B"

) -> str:
    """ Run Rag completion using Nebius models for resume optimizer."""
    try:
        llm = NebiusLLM(model=generative_model, api_key=os.getenv('NEBIUS_API_KEY'))
        embed_model = NebiusEmbedding(model_name=embedding_model, api_key=os.getenv('NEBIUS_API_KEY'))

        Settings.llm = llm
        Settings.embed_model = embed_model


        # Step 1: Analyze the resume
        analysis_prompt = f"""
        Analyze this resume in detail. Focus on:
        1. Key skills and expertise
        2. Professional experience and achievements
        3. Education and certifications
        4. Notable projects or accomplishments
        5. Career progression and gaps

        Provide a concise summary in bullet point.
        """
        index = VectorStoreIndex.from_documents(documents)
        resume_analysis=index.as_query_engine(similarity_top_k=5).query(analysis_prompt)

        # Step 2: Generate optimization suggestions
        optimization_prompt = f"""

        Based on the resume analysis and job requirements, provide specific, actionable improvements.
        ONLY return the final output. Do not include any reasoning, prompt text, or job description.


        Resume Analysis:
        {resume_analysis}

        Job Title: {job_title}
        Job Description: {job_description}

        Optimization Request: {query_text}

        Provide a direct, structured response in this exact format:

        ## Key Findings
        . [2-3 bullet points highlighting main alignment and gaps]

        ## Specific Improvements
        . [2-4 bullet points with concrete suggestions]
        . Each bullet should start with a strong action verb
        . Include specific examples where possible

        ## Action Items
        . [2-3 specific, immediate steps to take]
        . Each item should be clear and actionable

        Keep all the points concise and actionable. Do not include any thinking process or analysis.
        Do not repeat the job description or input prompt.

        """

        optimization_suggestions = index.as_query_engine(similarity_top_k=5).query(optimization_prompt)
        return str(optimization_suggestions)
    
    except Exception as e:
        raise 
