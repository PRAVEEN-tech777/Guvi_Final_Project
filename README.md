# Question Answering System with Hugging Face

## Project Overview
This project implements a Question Answering (QA) system using Hugging Face transformers. The system can understand questions and extract precise answers from given contexts, making it valuable for efficient information retrieval from large text volumes.

## Features
- Utilizes pre-trained transformer models from Hugging Face
- Supports GPU acceleration when available
- Includes evaluation metrics (Exact Match and F1 score)
- Provides a user-friendly interface using Gradio
- Follows PEP 8 coding standards
- Implements comprehensive error handling and logging

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/PRAVEEN-tech777/Guvi_Final_Project.git
cd Guvi_Final_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the QA System
```bash
python qa_system.py
```
This will start the Gradio interface, which you can access through your web browser.

### Example
```python
from qa_system import QASystem

qa = QASystem()
context = "The Amazon rainforest is one of the world's most biodiverse habitats."
question = "What is the Amazon rainforest known for?"
answer = qa.predict(context, question)
print(f"Answer: {answer}")
```

## Project Structure
```
├── qa_system.py        # Main implementation file
├── tests/              # Test files
│   └── test_qa.py      # Unit tests for QA system
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Testing
Run the tests using:
```bash
python -m pytest tests/
```

## Evaluation Metrics
The system uses standard QA evaluation metrics:
- Exact Match (EM): Measures exact answer matches
- F1 Score: Measures partial matches between predicted and actual answers

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Hugging Face for their transformer models and datasets
- The creators of the SQuAD dataset
