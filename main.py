# Importing required modules
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from happytransformer import HappyTextToText, TTSettings
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl.core import LengthSampler

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to format the response into key points
def format_response_as_key_points(response):
    sentences = response.split('. ')
    key_points = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence])
    return key_points

# Function to correct grammar mistakes
def correct_grammar(text):
    correction_result = happy_tt.generate_text(f"grammar: {text}", args=args)
    return correction_result.text

# Function to handle chat history and user input
def chat(chat_history, user_input):
    corrected_input = correct_grammar(user_input)
    bot_response = qa_chain.invoke({"query": corrected_input})
    bot_response = bot_response['result']
    formatted_response = format_response_as_key_points(bot_response)
    chat_history.append((user_input, formatted_response))
    return chat_history

# Initialize the HappyTextToText model for grammar correction
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)

# Define the checkpoint and tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Check if there's an updated model, otherwise use the base model
updated_model_path = "trained/updated_model"

try:
    if os.path.exists(updated_model_path):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            updated_model_path,
            device_map="auto",
            torch_dtype=torch.float32
        )
        print("Using the updated model.")
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float32
        )
        print("Using the base model.")
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Falling back to CPU-only mode.")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32
    )

# Create embeddings using SentenceTransformer
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the vectorstore database with embeddings
db = Chroma(persist_directory="trained", embedding_function=embeddings)

# Create a text generation pipeline
pipe = pipeline(
    'text2text-generation',
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

# Define the local language model
local_llm = HuggingFacePipeline(pipeline=pipe)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)

# Reinforcement Learning setup
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    updated_model_path if os.path.exists(updated_model_path) else checkpoint
).to(device)
ppo_config = PPOConfig(
    batch_size=1,
    learning_rate=1.41e-5,
    mini_batch_size=1,
    optimize_cuda_cache=True,
)
ppo_trainer = PPOTrainer(ppo_config, ppo_model, tokenizer=tokenizer)

# Function to generate response using PPO model
def generate_ppo_response(prompt):
    response_length_sampler = LengthSampler(100, 400)
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = ppo_model.generate(encoded_prompt, max_new_tokens=response_length_sampler())
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to save the model
def save_model():
    try:
        os.makedirs(updated_model_path, exist_ok=True)
        torch.save(ppo_model.state_dict(), os.path.join(updated_model_path, "model_state.pt"))
        tokenizer.save_pretrained(updated_model_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

# Function to check for answer in trained data and base model
def get_answer(query):
    # First, try to get an answer from the trained model
    response = qa_chain.invoke({"query": query})
    if response['result']:
        return response['result']

    # If no answer found, try the base model
    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"Error loading the base model: {e}")
        print("Falling back to CPU-only mode.")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32
        )

    base_pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    base_llm = HuggingFacePipeline(pipeline=base_pipe)
    base_qa_chain = RetrievalQA.from_chain_type(
        llm=base_llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        return_source_documents=True,
    )
    base_response = base_qa_chain.invoke({"query": query})
    return base_response['result']

# Main chat loop
chat_history = []
print("Welcome to the chat! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    # Correct user input grammar
    corrected_input = correct_grammar(user_input)

    # Generate response using PPO model
    ppo_response = generate_ppo_response(corrected_input)

    # Get the response from the trained data or base model
    original_response = get_answer(corrected_input)

    # Format both responses as key points
    ppo_response_formatted = format_response_as_key_points(ppo_response)
    original_response_formatted = format_response_as_key_points(original_response)

    # print(f"Saul (PPO):\n{ppo_response_formatted}")
    print(f"\nLawZen:\n{original_response_formatted}")

    # Optional feedback
    print("Was the PPO response better? (yes/no/skip): ", end="")
    feedback = input()

    # Update the model based on feedback (if provided)
    if feedback.lower() in ['yes', 'no']:
        try:
            reward = 1.0 if feedback.lower() == 'yes' else -1.0
            query_tensor = tokenizer.encode(corrected_input, return_tensors="pt").to(device)
            response_tensor = tokenizer.encode(ppo_response, return_tensors="pt").to(device)
            rewards = torch.tensor([reward]).to(device)

            # Update the model
            train_stats = ppo_trainer.step(
                [query_tensor],  # Wrap in a list
                [response_tensor],  # Wrap in a list
                [rewards]  # Wrap in a list
            )
            print("Model updated successfully.")
            print(f"Training stats: {train_stats}")

            # Save the model after updating
            save_model()
        except Exception as e:
            print(f"An error occurred while updating the model: {e}")
            print("Continuing without updating the model.")

    # Add to chat history
    chat_history.append((user_input, original_response_formatted))

    # Save chat history to a file
    try:
        with open("chat_history.txt", "a") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"LawZen: {original_response_formatted}\n\n")
        print("Chat history saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving chat history: {e}")

# Display the full conversation history
print("\nFull Conversation History:")
for prompt, response in chat_history:
    print(f"You: {prompt}")
    print(f"LawZen: {response}\n")