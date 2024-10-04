import json
import requests
from datetime import datetime

API_URL = "https://huggingface.co/api/daily_papers"
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")


def get_top_hf_papers(n: int, date: str = CURRENT_DATE) -> str:
    f"""
    Fetches the top N papers from the Hugging Face papers page based on the number of votes.

    Args:
    n (int): Number of top papers to fetch.
    date (str): The date of the papers to fetch, in the format "YYYY-MM-DD". Defaults to {date}.
    """
    top_papers = []
    try:
        response = requests.get(f"{API_URL}?date={date}&limit=50")
        response.raise_for_status()
        data = response.json()
        if not data:
            raise Exception("No papers found")

        paper_info = []
        for paper in data:
            title = paper.get("paper", {}).get("title", "Unknown")
            link = (
                f"https://huggingface.co/papers/{paper.get('paper', {}).get('id', '')}"
            )
            upvotes = paper.get("paper", {}).get("upvotes", 0)
            thumbnail = paper.get("thumbnail", "")

            authors = [
                author.get("name", "")
                for author in paper.get("paper", {}).get("authors", [])
            ]
            published_date = paper.get("paper", {}).get("publishedAt", "")
            summary = paper.get("paper", {}).get("summary", "")
            paper_info.append(
                {
                    "title": title,
                    "link": link,
                    "upvotes": upvotes,
                    "thumbnail": thumbnail,
                    "authors": authors,
                    "published_date": published_date,
                    "summary": summary,
                }
            )

        paper_info.sort(key=lambda x: x["upvotes"], reverse=True)
        top_papers = paper_info[:n]

    except requests.RequestException as e:
        print(f"Error fetching papers: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return json.dumps(top_papers, indent=2)


get_top_hf_papers_json = {
    "type": "function",
    "function": {
        "name": "get_top_hf_papers",
        "description": "Get the top N papers from the Hugging Face papers page based on the number of votes.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of top papers to fetch.",
                },
                "date": {
                    "type": "string",
                    "description": f'[Optional] Defaults to current date:"{CURRENT_DATE}". The date of the papers to fetch, in the format "YYYY-MM-DD".',
                },
            },
            "required": ["n"],
        },
    },
}

if __name__ == "__main__":
    print("Fetching top 5 papers from Hugging Face...", CURRENT_DATE)
    top_papers = get_top_hf_papers(5)
    print(top_papers)
