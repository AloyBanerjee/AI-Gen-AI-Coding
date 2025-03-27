import re
import streamlit as st
import pandas as pd
from docx import Document
import ollama
import json

def extract_text_from_docx(file):
    """
    Extracts raw text from a DOCX file.
    """
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_sections_with_ollama(text):
    """
    Uses an LLM (Ollama) to extract structured sections from text.
    """
    prompt = f"""
    Extract structured sections from the following document:
    
    {text}
    
    Return a JSON format like:
    {{
        "sections": [
            {{"section_name": "Introduction", "content": "..."}},
            {{"section_name": "Section 1: Safety Guidelines", "content": "..."}},
            ...
        ]
    }}
    """
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    try:
        return json.loads(response["message"]["content"])["sections"]
    except:
        return []

def compare_text(old_text, new_text):
    """
    Highlights differences between old and new text.
    """
    from difflib import Differ
    differ = Differ()
    diff_result = list(differ.compare(old_text.split(), new_text.split()))

    highlighted_diff = []
    for word in diff_result:
        if word.startswith("- "):  # Removed words
            highlighted_diff.append(f"<span style='background-color:#FFCCCC; color:red;'>{word[2:]}</span>")
        elif word.startswith("+ "):  # Added words
            highlighted_diff.append(f"<span style='background-color:#CCFFCC; color:green;'>{word[2:]}</span>")
        else:
            highlighted_diff.append(word[2:])

    return " ".join(highlighted_diff)

# Streamlit UI
st.set_page_config(page_title="Regulatory Document Comparison", layout="wide")
st.title("📑 Regulatory Document Comparison (Powered by Ollama)")

st.markdown("""
Upload two versions of a regulatory document (DOCX) to compare sections using an *open-source LLM (Mistral/Llama-3)*.
- ✅ *Old Document* 📄 → Before changes
- ✅ *New Document* 📄 → Updated version
- ✅ *Comparison Table* → Highlights differences in red (-removed) and green (+added)
""")

# Upload Files
col1, col2 = st.columns(2)
with col1:
    uploaded_old = st.file_uploader("📂 Upload Old Document (DOCX)", type=["docx"])
with col2:
    uploaded_new = st.file_uploader("📂 Upload New Document (DOCX)", type=["docx"])

compare=st.button("Analyze")

if uploaded_old and uploaded_new and compare:
    # Extract text
    old_text = extract_text_from_docx(uploaded_old)
    new_text = extract_text_from_docx(uploaded_new)

    # Extract sections using Ollama
    with st.spinner("🔄 Extracting sections with LLM..."):
        old_sections = extract_sections_with_ollama(old_text)
        new_sections = extract_sections_with_ollama(new_text)

    # Convert to dictionaries for comparison
    old_dict = {sec["section_name"]: sec["content"] for sec in old_sections}
    new_dict = {sec["section_name"]: sec["content"] for sec in new_sections}

    # Compare sections
    all_sections = sorted(set(old_dict.keys()).union(set(new_dict.keys())))
    comparison_data = []

    for section in all_sections:
        old_content = old_dict.get(section, "")
        new_content = new_dict.get(section, "")
        highlighted_diff = compare_text(old_content, new_content)

        if section in new_dict and section not in old_dict:
            comparison_data.append([section, "", new_content, highlighted_diff])  # Keep first column blank
        elif section in old_dict and section not in new_dict:
            comparison_data.append([section, old_content, "", highlighted_diff])
        else:
            comparison_data.append([section, old_content, new_content, highlighted_diff])

    # Create DataFrame
    df_comparison = pd.DataFrame(comparison_data, columns=["Section", "Old Document", "New Document ", "Differences"])

    # Display in Streamlit with color formatting
    st.markdown("### 📊 Section-wise Comparison")
    st.write(df_comparison.to_html(escape=False), unsafe_allow_html=True)
