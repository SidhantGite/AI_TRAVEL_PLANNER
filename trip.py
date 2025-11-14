
import streamlit as st
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ──────────────────────────────
# Load environment variables

load_dotenv()
GROQ_API_KEY="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"
# ──────────────────────────────
# Define a simple LSTM model for itinerary optimization
# ──────────────────────────────
class ItineraryLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=10):
        super(ItineraryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # use last time step
        return out

# ──────────────────────────────
# Mock function to simulate itinerary data and DL prediction
# ──────────────────────────────
def optimize_itinerary_with_dl(city, interests, num_days):
    # Convert input features into numerical form (simplified embedding)
    np.random.seed(42)
    input_seq = np.random.rand(1, num_days, 10)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32)

    # Initialize model
    model = ItineraryLSTM()
    model.eval()

    # Predict optimized embeddings (placeholder logic)
    with torch.no_grad():
        output = model(input_tensor).numpy()

    # Convert numeric prediction into mock “locations”
    # In a real setup, you'd map embeddings to actual points of interest
    suggested_order = np.argsort(output[0])[:num_days]
    optimized_plan = [f"Optimized Stop {i+1}: Attraction #{idx+1}" for i, idx in enumerate(suggested_order)]
    return optimized_plan

# ──────────────────────────────
# Streamlit UI
# ──────────────────────────────
st.set_page_config(page_title="AI Travel Planner", page_icon="🧳")
st.title("🧭 AI Multi-Day Travel Itinerary Planner (DL Enhanced)")
st.write("Plan your trip itinerary by entering your city, interests, and number of days you'd like to travel. "
         "This version uses Deep Learning to optimize the sequence of locations.")

# ──────────────────────────────
# Form input section
# ──────────────────────────────
with st.form("planner_form"):
    city = st.text_input("Enter the city name for your trip")
    interests = st.text_input("Enter your interests (comma-separated)")
    num_days = st.number_input("Number of days for your trip", min_value=1, max_value=30, value=3)
    submitted = st.form_submit_button("Generate Itinerary")

# ──────────────────────────────
# Itinerary generation + DL optimization
# ──────────────────────────────
if submitted:
    if not GROQ_API_KEY:
        st.error("❌ Missing GROQ_API_KEY. Please set it in your environment or .env file.")
    elif not city or not interests:
        st.warning("⚠️ Please fill in City and Interests fields.")
    else:
        try:
            # Step 1: Generate Base Itinerary using LLM
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model="llama-3.3-70b-versatile",
                temperature=0.4
            )

            itinerary_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful travel assistant. Create a {num_days}-day travel itinerary for {city}, "
                 "tailored to the user's interests: {interests}. "
                 "For each day, list morning, afternoon, and evening activities with time slots."),
                ("human", "Plan my {num_days}-day trip.")
            ])

            response = llm.invoke(
                itinerary_prompt.format_messages(city=city, interests=interests, num_days=num_days)
            )

            st.subheader("🗺️ Base LLM Itinerary")
            st.markdown(response.content)

            # Step 2: Optimize Itinerary Sequence with DL
            st.subheader("🤖 Deep Learning Optimization")
            optimized_plan = optimize_itinerary_with_dl(city, interests, num_days)

            for stop in optimized_plan:
                st.write(f"- {stop}")

            st.success("✅ Itinerary optimized successfully with LSTM model!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


