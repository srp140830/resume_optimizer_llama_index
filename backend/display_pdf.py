import streamlit as st
import base64
from PyPDF2 import PdfReader
import io

def display_pdf_preview(pdf_file):
    """ Display PDF preview in the sidebar."""

    try:
        st.sidebar.subheader("Resume Preview")
        base64_pdf=base64.b64encode(pdf_file.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.sidebar.error(f"Error previewing PDF: {str(e)}")
        return False
    

    
