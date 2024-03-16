# AI Document Assistant

The AI Document Assistant is a Python-based tool that utilizes artificial intelligence (AI) to assist users in understanding and extracting information from documents. Whether it's a PDF file, a text document, or a question about a specific topic, this assistant leverages various natural language processing (NLP) and machine learning techniques to provide relevant summaries, answers, and insights.

## Features
Summarize PDF documents
Answer questions based on document content
Perform web and Wikipedia searches for additional information
Support for various NLP models, including OpenAI's GPT-3.5
Interactive command-line interface (CLI) for easy interaction


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/SheriefAttia/AI-Document-Assistant.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY=your-api-key
    ```

## Usage

1. Run the application:

    ```bash
    python main.py
    ```

2. Follow the prompts to input the path to your PDF file, ask questions, and interact with the assistant.

## Configuration

- You can customize the behavior of the assistant by modifying the settings in the `config.py` file.

## Troubleshooting

If you encounter any issues or errors while using the AI Document Assistant, please check the following:

- Ensure that your OpenAI API key is correctly set as an environment variable.
- Make sure that the required Python dependencies are installed by running `pip install -r requirements.txt`.
- Verify that your PDF file is accessible and correctly formatted.

If the issue persists, feel free to open an [issue](https://github.com/SheriefAttia/AI-Document-Assistant/issues) on GitHub for assistance.

## Contributing

Contributions to the AI Document Assistant are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository and create a new branch.
2. Make your changes and test them thoroughly.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- OpenAI for providing access to the GPT-3.5 model
- PyMuPDF for PDF parsing functionality
- BeautifulSoup for web scraping
- Wikipedia API for accessing Wikipedia content
- Transformers and Sentence Transformers libraries for NLP functionality
