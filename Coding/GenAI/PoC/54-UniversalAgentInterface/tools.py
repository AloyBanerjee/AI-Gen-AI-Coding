from pathlib import Path
from typing import Optional

from agno.tools import Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools
from agno.tools.python import PythonTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.calculator import CalculatorTools
from agno.tools.duckdb import DuckDbTools
from agno.tools.exa import ExaTools
from agno.tools.csv_toolkit import CsvTools
from agno.tools.arxiv import ArxivTools
from agno.tools.email import EmailTools
# from agno.tools.github import GitHubTools
from agno.tools.googlecalendar import GoogleCalendarTools
from agno.tools.googlesheets import GoogleSheetsTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.jira import JiraTools
from agno.tools.pandas import PandasTools
# from agno.tools.pubmed import PubMedTools
# from agno.tools.reddit import RedditTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.youtube import YouTubeTools

cwd = Path(__file__).parent.resolve()
tmp_dir = cwd.joinpath("tmp")
tmp_dir.mkdir(exist_ok=True, parents=True)

def get_toolkit(tool_name: str) -> Optional[Toolkit]:
    if tool_name == "ddg_search":
        return DuckDuckGoTools(fixed_max_results=3)
    elif tool_name == "shell_tools":
        return ShellTools()
    elif tool_name == "file_tools":
        return FileTools(base_dir=cwd)
    elif tool_name == "python_tools":
        return PythonTools(base_dir=tmp_dir)
    elif tool_name == "yfinance":
        return YFinanceTools()
    elif tool_name == "calculator":
        return CalculatorTools(enable_all=True)
    elif tool_name == "duckdb":
        return DuckDbTools()
    # elif tool_name == "exa":
    #     return ExaTools(num_results=3)
    elif tool_name == "csv_tools":
        return CsvTools()
    elif tool_name == "arxiv":
        return ArxivTools()
    elif tool_name == "email":
        return EmailTools()
    # elif tool_name == "github":
    #     return GitHubTools()
    elif tool_name == "google_calendar":
        return GoogleCalendarTools()
    elif tool_name == "google_sheets":
        return GoogleSheetsTools()
    elif tool_name == "hackernews":
        return HackerNewsTools()
    elif tool_name == "jira":
        return JiraTools()
    elif tool_name == "pandas":
        return PandasTools()
    # elif tool_name == "pubmed":
    #     return PubMedTools()
    # elif tool_name == "reddit":
    #     return RedditTools()
    elif tool_name == "wikipedia":
        return WikipediaTools()
    elif tool_name == "youtube":
        return YouTubeTools()
    return None
