# Convert ANY website into an API using Firecrawl

This project lets you convert ANY website into an API using Firecrawl.
- [Firecrawl](https://www.firecrawl.dev/i/api) is used to scrape websites.
- Streamlit is used to create a web interface for the project.


---
## Setup and installations

**Get Firecrawl API Key**:
- Go to [Firecrawl](https://www.firecrawl.dev/i/api) and sign up for an account.
- Once you have an account, go to the API Key page and copy your API key.
- Paste your API key by creating a `.env` file as follows:

```
FIRECRAWL_API_KEY=your_api_key
```

**Install Dependencies**:
   Ensure you have Python 3.11 or later installed.
   ```bash
   pip install streamlit firecrawl firecrawl-py
   ```

---

## Run the project

Finally, run the project by running the following command:

```bash
streamlit run app.py
```



--