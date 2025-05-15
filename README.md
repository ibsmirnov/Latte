# LATTE ☕ — LLM-Assisted Topic/Thematic Analysis

The goal of this project is to assist with analysing the common themes or topics within large quantities of short texts while preserving researcher flexibility and control.

The project combines ideas from traditional computational approaches to topic analysis (e.g BERTopic), cluster annotations by LLMs, and visualisation allowing researchers to further validate and analyse results qualitatively.

For demonstration, we will use a sample of reddit post from two subreadits: AskMen and AskWomen.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setting up the environment

1. Clone the repository:
   ```
   git clone https://github.com/ibsmirnov/Latte.git
   cd Latte
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Preparation

1. Download the Reddit datasets from [Academic Torrents](https://academictorrents.com/) (search for "Reddit submissions" datasets)
2. Place the `AskMen_submissions.zst` and `AskWomen_submissions.zst` files in the `data/` directory
3. Run the sample creation script:
   ```
   python sample_creation.py
   ```
   This will generate a CSV file with samples from AskMen and AskWomen subreddits.

### Demo

Check out demo.ipynb to see how LATTE ☕ in practice