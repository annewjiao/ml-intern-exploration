import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

dataset_name = "stanfordnlp/imdb"
model_name = "distilbert-base-uncased"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
logger.info(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name, split='train')

# Load tokenizer and model
logger.info(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize dataset
logger.info("Tokenizing dataset")
dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_first_step=True,
    log_level='info',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    hub_model_id="fanc/imdb-binary-sentiment-distilbert",
    report_to="trackio",
    run_name="imdb-distilbert-finetune",
    trackio_space_id="fanc/ml-intern-imdb-distilbert"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=None # Placeholder for potential metrics
)

# Start training
logger.info("Starting training")
trainer.train()

# Push model to hub
trainer.push_to_hub()
logger.info("Training complete and model pushed to hub")