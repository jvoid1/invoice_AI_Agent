# Invoice AI Agent

An AI-powered Agent for web invoice automation that can navigate and interact with web pages using natural language instructions.

## Features

- Natural language web navigation
- Automated web interaction (clicking, typing, scrolling)
- Visual element recognition
- Support for complex web automation tasks as new tab opening

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <https://github.com/jvoid1/invoice_AI_Agent.git>
cd invoice_AI_Agent
```

2. Install using Poetry (recommended):
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install Playwright browsers
poetry run playwright install
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t invoice_AI_Agent .
```

## Configuration

1. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - Or set it as an environment variable:
     ```bash
     export OPENAI_API_KEY=your-api-key-here
     ```

## Running the Agent

### Using Poetry (Local)

```bash
poetry run python src/main.py
```

### Using Docker

```bash
docker run -it --env OPENAI_API_KEY=your-api-key invoice_AI_Agent
```

## Usage

The agent can handle various web invoice automation tasks through natural language commands/prompts and observations on each interaction with computer vision tools to navigate in the web with playwright.

### Example Commands

The agent understands natural language instructions like:
- "Search for specific button/text on a website"
- "Fill out forms with given data"
- "Navigate through multiple pages"
- "Extract information from web pages"

## Architecture

The agent uses:
- LangChain for the core AI functionality
- Playwright for web automation
- GPT-4 Vision for visual understanding
- LangGraph for workflow management
