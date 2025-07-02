import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import tempfile
import shutil

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from backend.display_pdf import display_pdf_preview
from backend.run_rag_completion import run_rag_completion


def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")

    # Initialize session states:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "documents" not in st.session_state:
       st.session_state.documents = None 

    # Header
    st.title("Resume Optimizer")


    # Sidebar for configuration
    with st.sidebar:

        generative_model = st.selectbox(
            "Generative Model", 
            ["Qwen/Qwen3-235B-A22B", "deepseek-ai/DeepSeek-V3"],
            index = 0
        )

        st.divider()

        # Resume Upload
        st.subheader ("Resume Upload")
        uploaded_pdf = st.file_uploader("Choose your resume (PDF)", type="pdf", accept_multiple_files=False)

        # Handle pdf upload and processing
        if uploaded_pdf is not None:
            if uploaded_pdf != st.session_state.current_pdf:
                st.session_state.current_pdf = uploaded_pdf


                try:
                    if not os.getenv('NEBIUS_API_KEY'):
                        st.error("Missing Nebius API Key")
                        st.stop

                    # Create temp directory for the uploaded pdf file
                    if st.session_state.temp_dir:
                        shutil.rmtree(st.session_state.temp_dir)
                    st.session_state.temp_dir = tempfile.mkdtemp()

                    # Save file to temp dir
                    file_path = os.path.join(st.session_state.temp_dir, uploaded_pdf.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())

                    with st.spinner("Loading Resume..."):
                        documents = SimpleDirectoryReader(st.session_state.temp_dir).load_data()
                        st.session_state.docs_loaded = True
                        st.session_state.documents = documents
                        st.success("✓ Resume loaded successfully")
                        display_pdf_preview(uploaded_pdf)



                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Main content area
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Job Information")
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description", height=200)

        st.subheader("Optimization Options")
        optimization_type=st.selectbox(
            "Select Optimization Type",
            [
                "ATS Keyword Optimizer",
                "Experience Section Enhancer",
                "Skills Hierarchy Creator",
                "Professional Summary Crafter",
                "Education Optimizer",
                "Technical Skills Showcase",
                "Career Gap Framing"
            ]
            )
        
        if st.button("Optimize Resume"):
            if not st.session_state.docs_loaded:
                st.error("Please upload your resume first")
                st.stop()

            if not job_title or not job_description:
                st.error(f"Please provide both job title and description")
                st.stop()

         # Generate optimization prompt based on selection
            prompts = {
                "ATS Keyword Optimizer": "Identify and optimize ATS keywords. Focus on exact matches and semantic variations from the job description.",
                "Experience Section Enhancer": "Enhance experience section to align with job requirements. Focus on quantifiable achievements.",
                "Skills Hierarchy Creator": "Organize skills based on job requirements. Identify gaps and development opportunities.",
                "Professional Summary Crafter": "Create a targeted professional summary highlighting relevant experience and skills.",
                "Education Optimizer": "Optimize education section to emphasize relevant qualifications for this position.",
                "Technical Skills Showcase": "Organize technical skills based on job requirements. Highlight key competencies.",
                "Career Gap Framing": "Address career gaps professionally. Focus on growth and relevant experience."
            }

            with st.spinner("Analyzing resume and generating suggestions..."):
                try:
                    response = run_rag_completion(
                        st.session_state.documents,
                        prompts[optimization_type],
                        job_title,
                        job_description,
                        "BAAI/bge-en-icl",
                        generative_model
                    )
                    response=response.replace("<think>", "").replace("</think>", "")
                    st.session_state.messages.append({"role":"assistant", "content": response})

                except Exception as e:
                    st.error(f"Error: {str(e)}")


    with col2:
        st.subheader("Optimization Results")
        if st.session_state.documents is not None:
            for message in st.session_state.messages:
                st.markdown(message["content"])

if __name__=="__main__":
    main()
