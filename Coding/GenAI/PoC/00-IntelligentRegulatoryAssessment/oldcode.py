
# Configure model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("🔍 Generating document structure...")
    # Prompt to return a list of structured nested sections/subsections
    retrieval_prompt = f'''
    You are a document structure analysis expert. Analyze the uploaded legal or regulatory document and extract its structure in JSON format.

    For each section in the document, extract:
    - Section title
    - A list of subsections where each has:
        - name: Title of the subsection
        - text: Text/content under the subsection

    Output format:
    [
        {{
            "document_section_name": "string",
            "document_subsection_name": [
                {{
                    "name": "string",
                    "text": "string"
                }},
                ...
            ]
        }},
        ...
    ]

    Here is the document content:
    {ocr_response}
    '''
    #print("🔍 Prompt for model:", retrieval_prompt)
    # Generate response
    response = model.generate_content(retrieval_prompt)
    print("🔍 Model response:", response.text)
    try:
        # Parse and validate structured JSON
        raw_output = json.loads(response.text)
        print("🔍 Parsed JSON:", raw_output)
        structured_output = [DocumentSectionExtraction(**item) for item in raw_output]
        return structured_output

    except (json.JSONDecodeError, ValidationError) as e:
        print("❌ Parsing failed:", e)
        print("🔎 Raw model output:", response.text)
        return []


vresults = []
    current_section = None
    current_subsection = None
    subsection_text_buffer = []
    subsections = []

    for page in ocr_response.pages:
        lines = page.markdown.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("# "):  # New section
                # Save previous section
                if current_section:
                    if current_subsection:
                        subsections.append(Subsection(name=current_subsection, text="\n".join(subsection_text_buffer)))
                    results.append(DocumentSectionExtraction(
                        document_section_name=current_section,
                        document_subsection_name=subsections
                    ))
                    subsections = []
                    subsection_text_buffer = []

                current_section = line[2:].strip()
                current_subsection = None

            elif line.startswith("## "):  # New subsection
                if current_subsection:
                    subsections.append(Subsection(name=current_subsection, text="\n".join(subsection_text_buffer)))
                current_subsection = line[3:].strip()
                subsection_text_buffer = []

            else:
                if current_subsection:
                    subsection_text_buffer.append(line)

        # Handle end of page
        if current_subsection:
            subsections.append(Subsection(name=current_subsection, text="\n".join(subsection_text_buffer)))
            subsection_text_buffer = []
        if current_section:
            results.append(DocumentSectionExtraction(
                document_section_name=current_section,
                document_subsection_name=subsections
            ))
            current_section = None
            current_subsection = None
            subsections = []
