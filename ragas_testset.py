from langchain.schema import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pdfplumber
import pandas as pd
from ragBalance import ragas_querying_question_with_score


from ragas.evaluation import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_pdf_document(pdf_path):
    """Extracts text from a PDF file and returns it as a Document."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    # Create and return a Document instance
    return Document(page_content=text, metadata={"source": pdf_path})

def generate_test_set() : 
  pdf_paths = ["uploaded_files/Kelas3_Pendidikan_Jasmani_Olahraga_dan_Kesehatan_Cleaned.pdf"]

  documents = [load_pdf_document(pdf_path) for pdf_path in pdf_paths]

  generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
  critic_llm = ChatOpenAI(model="gpt-4o-mini")
  embeddings = OpenAIEmbeddings()

  generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
  )

  # testset25 = generator.generate_with_langchain_docs(
  #   documents,
  #   test_size=25,
  #   distributions = {simple:0.5, reasoning:0.25, multi_context:0.25}
  # )

  testset10 = generator.generate_with_langchain_docs(
    documents,
    test_size=10,
    distributions = {simple:0.5, reasoning:0.25, multi_context:0.25}
  )

  # Convert testset to DataFrame
  testset10.to_pandas().to_excel("testset-gpt4.xlsx", index=False)

class TestSet:
    def __init__(self, df):
        self.df = df
        self.features = df.columns.tolist()

def process_and_load_testset(csv_path, indexname, namespace):
    try:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')

        # Verify the expected columns are present
        if 'question' not in df.columns or 'ground_truth' not in df.columns:
            raise ValueError("CSV file must contain 'question' and 'ground_truth' columns.")

        # Initialize lists for each key
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        # Loop through each row in the DataFrame
        for _, row in df.iterrows():
            question = row['question']
            ground_truth = row['ground_truth']

            answer, context = ragas_querying_question_with_score(question, indexname, namespace)
            
            # Ensure the values are strings and append them to the respective lists
            questions.append(question if isinstance(question, str) else str(question))
            answers.append(answer if isinstance(answer, str) else str(answer))
            contexts.append(context if isinstance(context, list) else [str(context)])
            ground_truths.append(ground_truth if isinstance(ground_truth, str) else str(ground_truth))
            
            # Construct the final DataFrame
            ragas_evaluate_df = pd.DataFrame({
                'question': questions,
                'ground_truths': ground_truths,
                'answer': answers,
                'contexts': contexts,
            })

        # Return the DataFrame
        # return ragas_evaluate_df

        # Return the custom TestSet object
        return ragas_evaluate_df

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


csv_path = "./testset-indo.csv"
indexname = "labira-edu-rag-dataset-tes-2"
namespace = "PJOK"

testset_dict = process_and_load_testset(csv_path, indexname, namespace)
# Save the DataFrame to an Excel file
print(testset_dict)
testset_dict.to_excel("processed_testset-indo-need-evaluate.xlsx", index=False)

# Ensure the 'evaluate' function is compatible with the DataFrame
# try:
#     result = evaluate(
#         testset_dict,
#         metrics=[
#             context_precision,
#             context_recall,
#             faithfulness,
#             answer_relevancy,
#         ],
#     )

#     # Convert the result to a DataFrame and save it to Excel
#     result_df = result.to_pandas()
#     result_df.to_excel("ragas-eval-gpt4.xlsx", index=False)
#     print("Results saved to 'ragas-eval-gpt4.xlsx'")

# except Exception as e:
#     print(f"An error occurred during evaluation: {e}")
