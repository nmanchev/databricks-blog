# Databricks notebook source
# MAGIC %md
# MAGIC # Speeding up LLM inference by using model quantization in Databricks
# MAGIC
# MAGIC This notebook serves as supplementary material to the [blog article link].
# MAGIC It includes the code snippets referenced in the blog post, providing a hands-on opportunity to explore and implement the concepts discussed.
# MAGIC
# MAGIC This notebook is an extension of the [Fine-tuning your LLM with Databricks Mosaic AI](https://notebooks.databricks.com/demos/llm-fine-tuning/index.html#) demo. It also leverages the acompanying NER dataset from the demo, which you can install by running:
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC %pip install dbdemos
# MAGIC
# MAGIC import dbdemos
# MAGIC dbdemos.install('llm-fine-tuning')
# MAGIC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the environment

# COMMAND ----------

# MAGIC %pip install -U databricks-genai databricks-sdk transformers optimum

# COMMAND ----------

# install AutoGPTQ from source (required for GPU config)
!mkdir /local_disk0/AutoGPTQ
!git clone https://github.com/PanQiWei/AutoGPTQ.git /local_disk0/AutoGPTQ
!pip install -vvv -e /local_disk0/AutoGPTQ/.

# COMMAND ----------

# restart the Python kernel to pick up the new libraries
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Build our prompt template to extract entities
# MAGIC
# MAGIC You need to set the `catalog` and `schema` variables to point to the locations of the NER tables (from the `llm-fine-tuning` demo).

# COMMAND ----------

catalog = "..."
schema = "..."
spark.sql(f"use catalog {catalog}")
spark.sql(f"use schema {schema}")

# COMMAND ----------

# DBTITLE 1,System Prompt Definition
system_prompt = """
### INSTRUCTIONS:
You are a medical and pharmaceutical expert. Your task is to identify pharmaceutical drug names from the provided input and list them accurately. Follow these guidelines:

1. Do not add any commentary or repeat the instructions.
2. Extract the names of pharmaceutical drugs mentioned.
3. Place the extracted names in a Python list format. Ensure the names are enclosed in square brackets and separated by commas and wrapped in double quotes, e.g. ["paracetamol", "ibuprofen"].
4. Maintain the order in which the drug names appear in the input.
5. Do not add any text before or after the list.
"""

# COMMAND ----------

# DBTITLE 1,Utility Functions
import json
import re
import pandas as pd

from pyspark.sql.functions import pandas_udf, to_json

# Extract the json array from the text, removing potential noise
def extract_json_array(text):
    # Use regex to find a JSON array within the text
    match = re.search(r"(\[.*?\])", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def get_current_cluster_id():
    import json

    return json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson()
    )["attributes"]["clusterId"]


@pandas_udf("array<struct<role:string, content:string>>")
def create_train_conv(sentence: pd.Series, entities: pd.Series) -> pd.Series:
    def build_message(s, e):
        # Default behavior with system prompt
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(s)},
            {"role": "assistant", "content": e},
        ]

    # Apply build_message to each pair of sentence and entity
    return pd.Series([build_message(s, e) for s, e in zip(sentence, entities)])


def create_test_conv(df):
    return df.apply(
        lambda row: [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["sentence"]},
        ],
        axis=1,
    )


def extract_entities(df, model):
    model.generation_config.update(temperature=0.1, max_new_tokens=100, max_length=None)
    results = model.predict(list(create_test_conv(df)))
    predictions = [p[0]["generated_text"][-1]["content"] for p in results]
    cleaned_predictions = [extract_json_array(p) for p in predictions]
    return predictions, cleaned_predictions

# COMMAND ----------

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_precision_recall(prediction, ground_truth):
    prediction_set = set([str(drug).lower() for drug in prediction])
    ground_truth_set = set([str(drug).lower() for drug in ground_truth])
    all_elements = prediction_set.union(ground_truth_set)

    # Convert sets to binary lists
    prediction_binary = [int(element in prediction_set) for element in all_elements]
    ground_truth_binary = [int(element in ground_truth_set) for element in all_elements]

    precision = precision_score(ground_truth_binary, prediction_binary)
    recall = recall_score(ground_truth_binary, prediction_binary)
    f1 = f1_score(ground_truth_binary, prediction_binary)

    return precision, recall, f1

# COMMAND ----------

def precision_recall_series(column_name):
    def inner(row):
        precision, recall, f1 = compute_precision_recall(
            row[column_name], row["human_annotated_entities"]
        )
        return pd.Series([precision, recall, f1], index=["precision", "recall", "f1"])

    return inner

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the data

# COMMAND ----------

from datasets import load_dataset
import pandas as pd

hf_dataset_name = "allenai/drug-combo-extraction"

# Specify a cache directory (e.g. Unity Catalog volume)
cache_dir = "..."

dataset_test = load_dataset(hf_dataset_name, split="test", cache_dir=cache_dir)

# Convert the dataset to a pandas DataFrame
df_test = pd.DataFrame(dataset_test)

# Extract the entities from the spans
df_test["human_annotated_entities"] = df_test["spans"].apply(
    lambda spans: [span["text"] for span in spans]
)

df_test = df_test[["sentence", "human_annotated_entities"]]

display(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the baseline model

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

base_model_name = "llama_v3_2_1b_instruct"
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
model_version = "2"

# COMMAND ----------

model_uri = f"models:/system.ai.{base_model_name}/{model_version}"
mdl_pipe = mlflow.transformers.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract entities with the baseline model

# COMMAND ----------

# Taking only a few examples from test set to collect benchmark metrics
from sklearn.model_selection import train_test_split

df_validation, df_test_small = train_test_split(df_test, test_size=0.2, random_state=42)

# COMMAND ----------

predictions, cleaned_predictions = extract_entities(df_test_small, mdl_pipe)
df_test_small["baseline_predictions"] = predictions
df_test_small["baseline_predictions_cleaned"] = cleaned_predictions
display(
    df_test_small[
        ["sentence", "baseline_predictions_cleaned", "human_annotated_entities"]
    ]
)

# COMMAND ----------

# MAGIC %md ## Evaluating our baseline model
# MAGIC
# MAGIC We can see that our model is extracting a good number of entities, but it also occasionally adds some random text after/before the inferences.
# MAGIC
# MAGIC ### Precision & recall for entity extraction
# MAGIC
# MAGIC We'll benchmark our model by computing its accuracy and recall. Let's compute these value for each sentence in our test dataset.

# COMMAND ----------

df_test_small[
    ["baseline_precision", "baseline_recall", "baseline_f1"]
] = df_test_small.apply(precision_recall_series("baseline_predictions_cleaned"), axis=1)
df_test_small[["baseline_precision", "baseline_recall", "baseline_f1"]].describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Fine-tuning our model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine tuning data preparation
# MAGIC
# MAGIC Before fine-tuning, we need to apply our prompt template to the samples in the training dataset, and extract the ground truth list of drugs into the list format we are targeting.
# MAGIC
# MAGIC We'll save this to our Databricks catalog as a table. Usually, this is part of a full Data Engineering pipeline.
# MAGIC
# MAGIC Remember that this step is key for your Fine Tuning, make sure your training dataset is of high quality.

# COMMAND ----------

dataset_train = load_dataset(hf_dataset_name, split="train", cache_dir=cache_dir)
df_train = pd.DataFrame(dataset_train)

# Convert the dataset to a pandas DataFrame
df_train = pd.DataFrame(df_train)

# Extract the entities from the spans
df_train["human_annotated_entities"] = df_train["spans"].apply(
    lambda spans: [span["text"] for span in spans]
)

df_train = df_train[["sentence", "human_annotated_entities"]]

df_train

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, to_json
import pandas as pd

# Assuming df_train is defined and correctly formatted as a Spark DataFrame with columns 'sentence' and 'entities'
training_dataset = spark.createDataFrame(df_train).withColumn(
    "human_annotated_entities", to_json("human_annotated_entities")
)

# Apply UDF, write to a table, and display it
training_dataset.select(
    create_train_conv("sentence", "human_annotated_entities").alias("messages")
).write.mode("overwrite").saveAsTable("ner_chat_completion_training_dataset")
display(spark.table("ner_chat_completion_training_dataset"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Prepare the eval dataset as well. We have the data available in `df_validation`

# COMMAND ----------

eval_dataset = spark.createDataFrame(df_validation).withColumn(
    "human_annotated_entities", to_json("human_annotated_entities")
)

# Apply UDF, write to a table, and display it
eval_dataset.select(
    create_train_conv("sentence", "human_annotated_entities").alias("messages")
).write.mode("overwrite").saveAsTable("ner_chat_completion_eval_dataset")
display(spark.table("ner_chat_completion_eval_dataset"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Fine-tuning
# MAGIC Once our data is ready, we can just call the fine tuning API

# COMMAND ----------

from databricks.model_training import foundation_model as fm

# Change the model name back to drug_extraction_ft after testing
registered_model_name = f"{catalog}.{schema}.drug_extraction_ft_" + re.sub(
    r"[^a-zA-Z0-9]", "_", str(base_model_name).lower()
)

# COMMAND ----------

# Create the fine tuning run
run = fm.create(
    data_prep_cluster_id=get_current_cluster_id(),  # Required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job. See ./_resources for more details
    model=base_model_path,
    train_data_path=f"{catalog}.{schema}.ner_chat_completion_training_dataset",
    eval_data_path=f"{catalog}.{schema}.ner_chat_completion_eval_dataset",
    task_type="CHAT_COMPLETION",
    register_to=registered_model_name,
    training_duration="10ep",  # Duration of the finetuning run, 10 epochs only to make it fast for the demo. Check the training run metrics to know when to stop it (when it reaches a plateau)
)
print(run)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Retrieve the fine tuned model and rerun the evaluation

# COMMAND ----------

ft_mdl_pipe = mlflow.transformers.load_model(f"models:/{registered_model_name}/1")

# COMMAND ----------

predictions, cleaned_predictions = extract_entities(df_test_small, ft_mdl_pipe)
df_test_small["ft_predictions"] = predictions
df_test_small["ft_predictions_cleaned"] = cleaned_predictions

# COMMAND ----------

df_test_small[["ft_precision", "ft_recall", "ft_f1"]] = df_test_small.apply(
    precision_recall_series("ft_predictions_cleaned"), axis=1
)
df_test_small[["ft_precision", "ft_recall", "ft_f1"]].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Quantize the fine tuned model

# COMMAND ----------

# Download the artifacts of the specified registered model version from MLflow
local_path = mlflow.artifacts.download_artifacts(f"models:/{registered_model_name}/1")

# COMMAND ----------

import torch

# Set the device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# COMMAND ----------

# Load the tokenizer and quantized model from the specified local path, load model to the GPU
tokenizer = AutoTokenizer.from_pretrained(
    local_path + "components/tokenizer", trust_remote_code=True
)

qnt_config = BaseQuantizeConfig(bits=4, group_size=128)
model = AutoGPTQForCausalLM.from_pretrained(
    local_path + "model", quantize_config=qnt_config
).to(device)

# COMMAND ----------

# Create a calibration dataset by extracting and tokenizing text messages from the fine tuning training data

calibration_json = (
    spark.table("ner_chat_completion_training_dataset").limit(3).toJSON().collect()
)
calibration_texts = [json.loads(c)["messages"] for c in calibration_json]

examples = []
for item in calibration_texts:
    text = tokenizer.apply_chat_template(
        item, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    examples.append(inputs)

# COMMAND ----------

# Quantize the model
model.quantize(examples)

# COMMAND ----------

!rm -r /local_disk0/quantized

# COMMAND ----------

!mkdir /local_disk0/quantized

# COMMAND ----------

# Save the quantized model to the local disk
model.save_quantized("/local_disk0/quantized/", use_safetensors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Retrieve the quantized model and rerun the evaluation

# COMMAND ----------

torch.cuda.empty_cache()

# COMMAND ----------

quantized_model = AutoGPTQForCausalLM.from_quantized("/local_disk0/quantized")

# COMMAND ----------

ft_qnt_mdl_pipe = TextGenerationPipeline(
    model=quantized_model.model, tokenizer=tokenizer
)

# COMMAND ----------

predictions, cleaned_predictions = extract_entities(df_test_small, ft_qnt_mdl_pipe)
df_test_small["qt_predictions"] = predictions
df_test_small["qt_predictions_cleaned"] = cleaned_predictions

# COMMAND ----------

df_test_small[["qt_precision", "qt_recall", "qt_f1"]] = df_test_small.apply(
    precision_recall_series("qt_predictions_cleaned"), axis=1
)
df_test_small[["qt_precision", "qt_recall", "qt_f1"]].describe()