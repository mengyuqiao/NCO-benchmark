from g4f.client import Client
import re
from tqdm import tqdm

gpt_group = ['gpt-4o-mini', 'gpt-4o-mini-tts','gpt-4.1-mini','gpt-4.1-nano',
            ]
llama_group = ["llama-4-maverick","llama-4-scout","llama-3.3-70b","llama-3.2-90b","llama-3.2-11b","llama-3.1-405b","llama-3.1-70b","llama-3.1-8b"][::-1]

deepseek_group = ['deepseek-v3',
    'deepseek-r1',
    'deepseek-r1-turbo',
    'deepseek-r1-distill-llama-70b',
    'deepseek-prover-v2-671b',
    'deepseek-r1-0528',
    'deepseek-r1-0528-turbo',
     'DeepSeek',]

def extract_questions_simple(file_path):
    """
    Extract questions as complete strings.
    
    Parameters:
    file_path (str): Path to the text file
    
    Returns:
    dict: Dictionary with question numbers as keys and full question text as values
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "Question XX" pattern
    pattern = r'Question (\d+)\n(.*?)(?=Question \d+\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    questions = []
    for _, question_text in matches:
        questions.append(question_text.strip())
        # print(question_text)
        # print('-'*100)

    return questions

filenames = [f"medical_questions_v{name}_nco.txt" for name in range(1,6)][::-1]

for file in filenames:
    questions = extract_questions_simple(file)
    answers = []
    for q in tqdm(questions):
        
        client = Client()
        for llm in deepseek_group:
            print("Calling", llm)
            try:
                response = client.chat.completions.create(
                    model=llm,
                    n=1,
                    messages=[
                        {"role": "user", "content": q + ""}],  # text only
                    timeout=60,
                    # Add any other necessary parameters
                ).choices[0].message.content.strip()
                
                print(response)
                answers.append(response)
                print("\n\n")
                break
            except: pass
    # print(answers)