import os


project_path = os.path.dirname(os.path.realpath(__file__))


# Model Path
roberta_large_path = os.path.join(project_path, "saved_model/roberta_large")

# Dataset Path
tacred_path = os.path.join(project_path, "saved_data/tacred")
tacrev_path = os.path.join(project_path, "saved_data/tacrev")
retacred_path = os.path.join(project_path, "saved_data/retacred")
semeval_path = os.path.join(project_path, "saved_data/semeval")

# Output Path
saved_output_path = os.path.join(project_path, "saved_output")
