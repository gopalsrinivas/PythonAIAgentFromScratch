from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Save research output to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Data successfully saved to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


# Define tools
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = Tool(
    name="wikipedia_search",
    func=api_wrapper.run,
    description="Search Wikipedia for information.",
)


# Get all tools
def get_tools():
    return [search_tool, wiki_tool, save_tool]
