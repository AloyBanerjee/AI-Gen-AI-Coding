# Multi-Agent AI Researcher

## Overview
The **Multi-Agent AI Researcher** is a Streamlit-based application that utilizes **HackerNews tools** to conduct comprehensive research and analysis on HackerNews stories and user profiles. This tool is designed for researchers, analysts, and tech enthusiasts, providing a collaborative AI-powered system for extracting insights, identifying trends, and generating meaningful content from HackerNews data.

---

## Key Features
- **Collaborative AI Agents**: Multiple AI assistants work together to analyze stories and user profiles.
- **GroqChat Integration**: Leverages the **deepseek-r1-distill-llama-70b** model to deliver precise and insightful responses.
- **HackerNews Analysis Tools**: Extracts and interprets information from stories, profiles, and emerging trends.
- **Customizable System Prompt**: Ensures tailored, in-depth, and actionable responses.
- **User-Friendly Interface**: A simple and intuitive Streamlit-based UI for seamless user interaction.

---

## Installation Guide

### Prerequisites
Ensure Python 3.8 or later is installed on your system.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-agent-researcher.git
   cd multi-agent-researcher
   ```
2. Install dependencies:
   ```bash
   pip install phi streamlit
   ```
3. Set up the Groq API key:
   - Input the key in the designated field upon launching the app.

---

## How to Use

### Running the Application
1. Start the app:
   ```bash
   streamlit run app.py
   ```
2. **Interacting with the UI**:
   - Enter your **Groq API Key**.
   - Provide a research query regarding HackerNews stories or users.
   - Click **Generate Insights** to receive a comprehensive analysis.

---

## Technical Breakdown

### AI Assistant Modules
- **HackerNews Story Analyzer**:
  - Examines trending HackerNews stories.
  - Identifies key topics, trends, and significance.
- **HackerNews User Investigator**:
  - Reviews user profiles, activities, and contributions within the community.
- **HackerNews Insights Engine**:
  - Uses **GroqChat** to provide a holistic analysis by combining story and user data.

### System Prompt Optimizations
The structured system prompt ensures:
- Context-rich, data-driven insights.
- Interpretation of trends in a broader technological and societal landscape.
- Suggestions for further areas of exploration.

### Example Workflow
1. **User Query**: "What are the latest AI trends from HackerNews stories?"
2. **System Response**:
   - Analyzes top AI-related stories.
   - Identifies recurring themes and discussion frequency.
   - Explores implications for the broader tech industry.

---

## UI Previews
- **Home Screen**: Fields for API key input and research queries.
- **Insights Display**: AI-generated analysis presented in an organized markdown format.

---

## Planned Enhancements
- Integrate visualization tools for trend analysis (charts, graphs, etc.).
- Extend functionality to cover additional sources such as Reddit and Medium.
- Enable multi-query sessions with history tracking.

---

