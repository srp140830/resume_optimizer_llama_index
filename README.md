A powerful AI-powered resume optimization tool that helps enhance your resumes based on specific job requirements. Built with Streamlit.
This is inspired from  https://github.com/Arindam200/awesome-ai-apps/blob/main/rag_apps/resume_optimizer repo. 

This application provides target suggestions to improve resume's effectiveness


## Features

- **PDF Resume Processing**: Upload and analyze your resume in PDF format
- **Job-Specific Optimization**: Get tailored suggestions based on job title and description
- **Multiple Optimization Types**:
  - ATS Keyword Optimizer
  - Experience Section Enhancer
  - Skills Hierarchy Creator
  - Professional Summary Crafter
  - Education Optimizer
  - Technical Skills Showcase
  - Career Gap Framing
- **AI-Powered Analysis**: Leverages advanced language models for intelligent suggestions

## Prerequisites

- Python 3.10 or higher
- PDF resume file

## Installation

1. Clone the repository:

```bash
git https://github.com/srp140830/resume_optimizer_llama_index.git
cd resume_optimizer_llama_index/

```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Nebius API key:

```
NEBIUS_API_KEY=your_api_key_here
```

## Usage

1. Start the application:

```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. In the application:
   - Upload your resume (PDF format)
   - Enter the job title
   - Provide the job description
   - Select the type of optimization you want
   - Click "Optimize Resume" to get AI-powered suggestions




