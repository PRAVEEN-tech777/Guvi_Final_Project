!pip install torch transformers datasets evaluate gradio tqdm

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import gradio as gr
import logging




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    """
    A Question Answering system using transformer models.

    Attributes:
        device (torch.device): The device (CPU/GPU) where the model will run
        tokenizer: The tokenizer for processing input text
        model: The transformer model for question answering
    """

    def __init__(self, model_name="distilbert-base-cased-distilled-squad"):
        """
        Initialize the QA system.

        Args:
            model_name (str): The name of the pre-trained model to use
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing QA system: {str(e)}")
            raise

    def predict(self, context, question):
        """
        Predict the answer to a question given a context.

        Args:
            context (str): The context text
            question (str): The question to answer

        Returns:
            str: The predicted answer

        Raises:
            ValueError: If context or question is empty
        """
        if not context or not question:
            raise ValueError("Context and question cannot be empty")

        try:
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)

            answer = self.tokenizer.decode(
                inputs.input_ids[0][answer_start:answer_end + 1],
                skip_special_tokens=True
            )

            return answer
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

class DataProcessor:
    """Handles dataset loading and preparation."""

    def __init__(self, dataset_name="squad"):
        """
        Initialize the data processor.

        Args:
            dataset_name (str): The name of the dataset to load
        """
        try:
            self.dataset = load_dataset(dataset_name)
            logger.info(f"Loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def prepare_data(self):
        """
        Prepare the dataset for training and validation.

        Returns:
            tuple: (train_dataset, validation_dataset)
        """
        return self.dataset["train"], self.dataset["validation"]

class ModelTrainer:
    def __init__(self, qa_system, data_processor):
        self.qa_system = qa_system
        self.data_processor = data_processor
        self.metric = evaluate.load("squad")

    def evaluate(self, dataset):
        predictions = []
        references = []

        for example in tqdm(dataset):
            prediction = self.qa_system.predict(example['context'], example['question'])
            predictions.append({"prediction_text": prediction, "id": str(len(predictions))})
            references.append({"answers": {"text": [example['answers']['text'][0]],
                                          "answer_start": [example['answers']['answer_start'][0]]},
                              "id": str(len(references))})

        results = self.metric.compute(predictions=predictions, references=references)
        return results

def create_gradio_interface(qa_system):
    def qa_interface(context, question):
        return qa_system.predict(context, question)

    iface = gr.Interface(
        fn=qa_interface,
        inputs=[
            gr.Textbox(lines=5, label="Context"),
            gr.Textbox(lines=2, label="Question")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Question Answering System",
        description="Enter a context and ask a question about it."
    )
    return iface


def main():
    """Main function to run the QA system."""
    try:
        qa_system = QASystem()
        data_processor = DataProcessor()
        trainer = ModelTrainer(qa_system, data_processor)

        train_dataset, validation_dataset = data_processor.prepare_data()

        logger.info("Evaluating model on validation dataset...")
        eval_results = trainer.evaluate(validation_dataset.select(range(100)))
        logger.info(f"Evaluation results: {eval_results}")

        iface = create_gradio_interface(qa_system)
        iface.launch()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
