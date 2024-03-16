import os
import re
import requests
from transformers import pipeline
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import wikipediaapi
import spacy
from sentence_transformers import SentenceTransformer, util
import random
import requests

nlp_sim = spacy.load("en_core_web_sm")
# Configure environment for minimal logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def clean_text(text):
    """Simplifies PDF text by removing headers, footers, and sections likely to be irrelevant."""
    text = re.sub(r'Page \d+ of \d+', '', text)  # Remove pagination
    text = re.sub(r'\n\s*\n', '\n', text)  # Collapse multiple newlines
    return text.strip()


def query_openai_gpt(prompt, api_key):
    """Sends a prompt to the OpenAI API and returns the text result or None on error."""
    url = "https://api.openai.com/v1/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    json_data = {
        "model": "gpt-3.5-turbo",  # Consider updating to the latest model as necessary
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()  # This will raise an exception for 4XX/5XX errors
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None


def process_pdf_text(file_path):
    """Extracts and prepares text from a PDF for further processing."""
    full_text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            full_text.append(clean_text(page.get_text()))
    return ' '.join(full_text)


def refine_answer(answer, question):
    """Enhances answers with additional information based on the question's context."""
    enhancements = {
        "python": "For more detailed information, refer to the official Python documentation.",
        "machine learning": "Consider exploring more on TensorFlow or PyTorch for practical applications."
    }
    for key, info in enhancements.items():
        if key in question.lower():
            answer += f" {info}"
    return answer

def search_web(query):
    """Performs a web search and returns the most relevant snippet."""
    try:
        response = requests.get(f"https://www.google.com/search?q={query}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        snippet = soup.find('div', class_='BNeawe').text
        return snippet
    except Exception as e:
        print(f"Web search failed: {e}")
        return None


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the most relevant article."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        url = f"https://en.wikipedia.org/wiki/Special:Search/{query}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4XX/5XX errors
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='mw-search-result-heading')
        if search_results:
            first_result_link = search_results[0].find('a')['href']
            page_url = f"https://en.wikipedia.org{first_result_link}"
            page_response = requests.get(page_url, headers=headers)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            content_paragraphs = page_soup.find_all('p')
            content = ' '.join([p.get_text() for p in content_paragraphs])
            return content[:500]  # Return the first 500 characters of the page content
        else:
            return None
    except requests.RequestException as e:
        print(f"Wikipedia search failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during Wikipedia search: {e}")
        return None

def local_fallback_nlp(full_text, question=None):
    try:
        nlp = pipeline("question-answering" if question else "summarization")
        model_max_length = nlp.tokenizer.model_max_length

        if len(full_text) > model_max_length:
            print("Full text too long for direct processing, using chunking.")

            # Chunking logic
            max_chunk_length = 8192  # Adjust this
            chunks = [full_text[i:i + max_chunk_length] for i in range(0, len(full_text), max_chunk_length)]

            processed_chunks = []  # Store results from each chunk
            for chunk in chunks:
                if question:
                    processed_text = nlp(question=question, context=chunk)['answer']
                else:
                    processed_text = nlp(chunk, max_length=1024, min_length=30, do_sample=False)[0]['summary_text']
                processed_chunks.append(processed_text)

            return ' '.join(processed_chunks).strip()

        else:
            # Attempt with the entire PDF text
            processed_text = nlp(question=question, context=full_text)['answer'] if question else \
                nlp(full_text, max_length=1024, min_length=30, do_sample=False)[0]['summary_text']
            return processed_text.strip()

    except Exception as e:
        print(f"Local processing failed: {e}")
        return None


def smart_chunking(full_text, max_chunk_size=8192, overlap_size=1024):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(full_text)

    chunks = []
    start_index = 0

    while start_index < len(full_text):
        end_index = min(start_index + max_chunk_size, len(full_text))

        # Find a potential chunk boundary near the end_index (prefer sentence breaks)
        chunk_boundary = end_index
        for sent in doc.sents:
            if sent.start > end_index:
                break  # We've passed the current chunk limit
            if sent.end >= end_index:
                chunk_boundary = sent.end

        chunk = full_text[start_index:chunk_boundary]
        chunks.append(chunk)

        # Calculate the next starting point with overlap
        start_index = max(0, chunk_boundary - overlap_size)

    return chunks


def semantic_similarity(question, answer):
    model = SentenceTransformer('paraphrase-distilroberta-base-v2')  # Or another suitable model
    embeddings = model.encode([question, answer])
    cos_sim = util.cos_sim(embeddings[0], embeddings[1])
    return cos_sim.item()  # Returns a similarity score


def similarity(question, answer):
    """Calculates a basic similarity score between the question and answer."""
    question_tokens = nlp_sim(question)
    answer_tokens = nlp_sim(answer)
    return question_tokens.similarity(answer_tokens)


def validate_answers(answers, question):
    """Validate answers based on relevance to the question."""
    if not answers:
        return None  # No answers provided

    # Filter out answers that don't contain any relevant information
    filtered_answers = [answer for answer in answers if answer.strip()]

    if not filtered_answers:
        return None  # No relevant answers found

    # Return a randomly selected relevant answer to add variety
    return random.choice(filtered_answers)


def formulate_enhanced_query(query):
    """Enhance the query based on its content."""
    if not query:
        return None  # No query provided

    # Add additional keywords or context to the query based on common patterns
    enhanced_query = query + " best practices"  # Example: Adding "best practices" for improvement

    return enhanced_query


def extract_content_from_wikipedia(page):
    """Extract relevant content from a Wikipedia page."""
    if not page:
        return None  # No Wikipedia page provided

    # Extract the main body text from the Wikipedia page
    content = page.text
    return content


def main():
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")  # Encourage use of environment variables

    file_path = input("Please enter the path to your PDF file: ")
    full_text = process_pdf_text(file_path)  # Store the returned value

    # Attempt summarization with OpenAI GPT or fall back to local processing
    summary_prompt = "Please summarize this text:\n" + full_text
    summary = query_openai_gpt(summary_prompt, api_key) or local_fallback_nlp(full_text)
    print("Document Summary:", summary)

    if summary is not None:
        # Interactive question-answering loop
        while True:
            question = input("\nAsk a question about the document or type 'exit' to quit: ").strip().lower()
            if question == 'exit':
                break  # Exit the loop when the user types 'exit'

            if not question:
                print("Please enter a question.")
                continue

            answer = query_openai_gpt(question, api_key) or local_fallback_nlp(full_text, question)

            if not answer or is_answer_unsatisfactory(answer, question):
                user_input = input("Do you want to search the web or Wikipedia for more relevant answers? (web/wiki/exit): ").strip().lower()
                if user_input == 'web':
                    answer = search_web(question)
                elif user_input == 'wiki':
                    answer = search_wikipedia(question)
                elif user_input == 'exit':
                    break
                else:
                    print("Invalid option. Please choose 'web', 'wiki', or 'exit'.")

            print("Answer:", refine_answer(answer, question))
    else:
        print("Unable to generate summary. Exiting.")

def is_answer_unsatisfactory(answer, user_input):
    # Check if the answer is below a certain length threshold
    if len(answer) < 30:  # Adjust the threshold as needed
        return True

    # Check if the user input contains certain keywords indicating dissatisfaction
    poor_quality_keywords = ["i don't know", "not sure", "sorry, i couldn't find relevant information", "no", "wrong answer", "search"]
    user_input_lower = user_input.lower().strip()  # Convert to lowercase and strip whitespace

    # Check if any keyword is present in the user input
    for keyword in poor_quality_keywords:
        if keyword.lower() in user_input_lower:
            return True

    return False

if __name__ == "__main__":
    main()
