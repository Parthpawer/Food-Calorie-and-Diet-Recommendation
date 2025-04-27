import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import google.generativeai as genai
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Food Calorie Detection & Diet Recommendation",
    page_icon="🥗",
    layout="wide"
)

# Configuration for Gemini API
def configure_gemini_api():
    # Using direct API key
    api_key = "Your_API_KEY"
    genai.configure(api_key=api_key)
    return True

# Load the diet recommendation models
@st.cache_resource
def load_models():
    try:
        # Load the diet recommendation models from the repo
        model = pickle.load(open('diet_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the model files are in the correct location.")
        return None, None

# Function to detect food and calories using Gemini API
def detect_food_calories(image_data, model_option="gemini-1.5-flash"):
    try:
        # Initialize Gemini model based on user selection
        model = genai.GenerativeModel(model_option)
        
        # Convert image to format required by Gemini
        image = Image.open(io.BytesIO(image_data))
        
        # Create prompt for Gemini - improved for better structured output
        prompt = """
        Analyze this food image and provide the following information.
        
        1. Food items in the image (list all visible food items)
        2. Estimated calories per serving (single number without notes)
        3. Approximate macronutrients in grams (single numbers without notes):
           - Protein 
           - Carbohydrates
           - Fat
        4. Common allergens present
        
        Format your response as a valid JSON object with this exact structure:
        {
          "food_items": ["item1", "item2", ...],
          "calories": ,
          "protein_g": ,
          "carbs_g": ,
          "fat_g": ,
          "allergens": ["allergen1", "allergen2", ...]
        }
        
        Important: Values for calories, protein_g, carbs_g, and fat_g must be single numerical values only, no additional text or notes.
        """
        
        # Generate response from Gemini
        response = model.generate_content([prompt, image])
        
        # Extract JSON from response
        import json
        import re
        
        # Find JSON in response
        response_text = response.text
        
        # First try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({\s*".*?"\s*:.*?})\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to extract JSON without markdown code blocks - looking for the complete JSON structure
            json_match = re.search(r'({(?:\s*"[^"]+"\s*:\s*(?:"[^"]*"|\d+|\[(?:"[^"]*",?\s*)*\]|\{[^}]*\}),?\s*)+})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("Could not extract JSON from Gemini response")
        
        # Clean the string before parsing
        json_str = re.sub(r'("[^"]*"):\s*({[^}]*})', r'\1: "\2"', json_str)  # Convert nested objects to strings
        
        try:
            result = json.loads(json_str)
            
            # Ensure all required fields exist and have the correct type
            if "food_items" not in result:
                result["food_items"] = ["Unknown food"]
            elif isinstance(result["food_items"], str):
                result["food_items"] = [result["food_items"]]
                
            # Convert to simple numeric values if they're complex objects
            for field in ["calories", "protein_g", "carbs_g", "fat_g"]:
                if field not in result:
                    result[field] = 0
                elif isinstance(result[field], dict) and "estimate" in result[field]:
                    result[field] = result[field]["estimate"]
                elif not isinstance(result[field], (int, float)):
                    try:
                        result[field] = float(str(result[field]).split()[0])
                    except:
                        result[field] = 0
                        
            # Ensure allergens is a list
            if "allergens" not in result:
                result["allergens"] = []
            elif isinstance(result["allergens"], str):
                result["allergens"] = [result["allergens"]]
                
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
            st.code(json_str)  # Show the problematic JSON
            raise ValueError(f"Invalid JSON format: {e}")
        return result
    
    except Exception as e:
        st.error(f"Error detecting food: {str(e)}")
        return None

# Function to recommend diet based on user inputs
def recommend_diet(age, weight, height, gender, activity_level, medical_condition):
    try:
        model, scaler = load_models()
        if model is None or scaler is None:
            return None
        
        # Gender encoding (assuming 0 for Male and 1 for Female as per original model)
        gender_code = 1 if gender == "Female" else 0
        
        # Activity level encoding (assuming 1-5 scale as per original model)
        activity_mapping = {
            "Sedentary": 1,
            "Lightly Active": 2,
            "Moderately Active": 3,
            "Very Active": 4,
            "Extra Active": 5
        }
        activity_code = activity_mapping.get(activity_level, 3)
        
        # Medical condition encoding (assuming these are the classes from the original model)
        # Modify according to the actual model's classes
        condition_mapping = {
            "None": 0,
            "Diabetes": 1,
            "Cholesterol": 2,
            "Blood Pressure": 3,
            "PCOD": 4
        }
        condition_code = condition_mapping.get(medical_condition, 0)
        
        # Calculate BMI
        height_m = height / 100  # Convert height from cm to meters
        bmi = weight / (height_m ** 2)
        
        # Calculate BMR using Harris-Benedict Equation
        if gender == "Male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        # Calculate daily calorie needs
        activity_factors = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extra Active": 1.9
        }
        daily_calories = bmr * activity_factors.get(activity_level, 1.55)
        
        # Prepare input for model prediction
        input_data = np.array([[age, weight, height, gender_code, bmi, activity_code, condition_code]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        diet_plan_id = model.predict(input_scaled)[0]
        
        # Map diet plan ID to recommendations (update according to model's actual predictions)
        diet_plans = {
            0: {
                'name': 'Weight Loss Diet',
                'description': 'A calorie-restricted diet focused on lean proteins and vegetables.',
                'daily_calories': daily_calories * 0.8,  # 20% deficit
                'recommendations': [
                    'Breakfast: Egg white omelet with vegetables',
                    'Lunch: Grilled chicken salad',
                    'Dinner: Baked fish with steamed vegetables',
                    'Snacks: Greek yogurt, nuts, or fruits'
                ]
            },
            1: {
                'name': 'Balanced Diet',
                'description': 'A well-balanced diet with moderate calories.',
                'daily_calories': daily_calories,
                'recommendations': [
                    'Breakfast: Oatmeal with berries and nuts',
                    'Lunch: Quinoa bowl with vegetables and lean protein',
                    'Dinner: Grilled protein with vegetables and whole grains',
                    'Snacks: Fruit, nuts, or yogurt'
                ]
            },
            2: {
                'name': 'Weight Gain Diet',
                'description': 'A calorie-surplus diet focused on healthy protein and fats.',
                'daily_calories': daily_calories * 1.2,  # 20% surplus
                'recommendations': [
                    'Breakfast: Protein smoothie with oats and nut butter',
                    'Lunch: Chicken or tofu with rice and vegetables',
                    'Dinner: Salmon with sweet potatoes and vegetables',
                    'Snacks: Nuts, avocado toast, protein shakes'
                ]
            },
            3: {
                'name': 'Diabetic Diet',
                'description': 'A diet focused on managing blood sugar levels.',
                'daily_calories': daily_calories * 0.9,  # 10% deficit
                'recommendations': [
                    'Breakfast: Steel-cut oats with cinnamon and berries',
                    'Lunch: Lentil soup with a side salad',
                    'Dinner: Grilled fish with non-starchy vegetables',
                    'Snacks: Nuts, vegetable sticks with hummus'
                ]
            },
            4: {
                'name': 'Heart-Healthy Diet',
                'description': 'A diet focused on heart health and cholesterol management.',
                'daily_calories': daily_calories * 0.9,  # 10% deficit
                'recommendations': [
                    'Breakfast: Overnight oats with flaxseeds',
                    'Lunch: Vegetable soup with whole grain bread',
                    'Dinner: Baked skinless chicken with vegetables',
                    'Snacks: Fresh fruits, unsalted nuts'
                ]
            }
        }
        
        return {
            'diet_plan_id': int(diet_plan_id),
            'plan': diet_plans.get(int(diet_plan_id), diet_plans[1]),
            'bmr': bmr,
            'daily_calories': daily_calories,
            'bmi': bmi
        }
        
    except Exception as e:
        st.error(f"Error recommending diet: {str(e)}")
        return None

# Main app
def main():
    st.title("🥗 Food Calorie Detection & Diet Recommendation System")
    
    # Check if Gemini API is configured
    api_configured = configure_gemini_api()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Food Calorie Detection", "Diet Recommendation"])
    
    if page == "Home":
        st.header("Welcome to the Food Calorie Detection & Diet Recommendation System")
        st.write("""
        This application helps you:
        - Detect calories in food through image analysis
        - Get personalized diet recommendations based on your health profile
        
        Use the sidebar to navigate between features.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Food Calorie Detection")
            st.write("Upload an image of your food to get calorie information.")
            st.image("https://via.placeholder.com/300x200.png?text=Food+Analysis", use_column_width=True)
            if st.button("Try Food Calorie Detection"):
                st.session_state.page = "Food Calorie Detection"
                st.experimental_rerun()
        
        with col2:
            st.subheader("Diet Recommendation")
            st.write("Enter your health information to get personalized diet recommendations.")
            st.image("https://via.placeholder.com/300x200.png?text=Diet+Recommendation", use_column_width=True)
            if st.button("Try Diet Recommendation"):
                st.session_state.page = "Diet Recommendation"
                st.experimental_rerun()
    
    elif page == "Food Calorie Detection":
        st.header("Food Calorie Detection")
        
        if not api_configured:
            st.warning("Please configure the Gemini API to use this feature.")
            return
        
        uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="Uploaded Food Image", use_column_width=True)
            
                            # Add model selection
            model_option = st.selectbox(
                "Select Gemini model",
                ["gemini-1.5-flash", "gemini-pro-vision"],
                index=0
            )
            
            # Process the image when button is clicked
            if st.button("Analyze Food"):
                with st.spinner("Analyzing food image..."):
                    # Get food analysis from Gemini
                    food_info = detect_food_calories(image_bytes, model_option)
                    
                    if food_info:
                        # Display results in a nice format
                        st.success("Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Food Identified")
                            food_items = food_info.get("food_items", ["Unknown food"])
                            for i, item in enumerate(food_items):
                                st.write(f"• {item}")
                            
                            st.subheader("Calorie Information")
                            st.metric("Calories per serving", f"{food_info.get('calories', 'N/A')} kcal")
                        
                        with col2:
                            st.subheader("Macronutrients")
                            st.metric("Protein", f"{food_info.get('protein_g', 'N/A')} g")
                            st.metric("Carbohydrates", f"{food_info.get('carbs_g', 'N/A')} g")
                            st.metric("Fat", f"{food_info.get('fat_g', 'N/A')} g")
                        
                        st.subheader("Potential Allergens")
                        allergens = food_info.get("allergens", [])
                        if allergens and len(allergens) > 0:
                            for allergen in allergens:
                                st.write(f"• {allergen}")
                        else:
                            st.write("No common allergens detected")
                            
                        # Add debug option to show raw response
                        # with st.expander("Show raw API response"):
                        #     st.code(response.text)
                    else:
                        st.error("Could not analyze the food image. Please try again with a clearer image.")
    
    elif page == "Diet Recommendation":
        st.header("Personalized Diet Recommendation")
        
        # User input form
        with st.form("diet_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=15, max_value=120, value=30)
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                activity_level = st.selectbox(
                    "Activity Level",
                    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"],
                    index=2
                )
                medical_condition = st.selectbox(
                    "Medical Condition",
                    ["None", "Diabetes", "Cholesterol", "Blood Pressure", "PCOD"]
                )
            
            submit_button = st.form_submit_button("Get Diet Recommendation")
        
        if submit_button:
            with st.spinner("Generating personalized diet recommendation..."):
                recommendation = recommend_diet(age, weight, height, gender, activity_level, medical_condition)
                
                if recommendation:
                    st.success("Diet recommendation generated!")
                    
                    # Display health metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BMI", f"{recommendation['bmi']:.1f}")
                        bmi_category = ""
                        if recommendation['bmi'] < 18.5:
                            bmi_category = "Underweight"
                        elif recommendation['bmi'] < 25:
                            bmi_category = "Normal weight"
                        elif recommendation['bmi'] < 30:
                            bmi_category = "Overweight"
                        else:
                            bmi_category = "Obesity"
                        st.write(f"Category: {bmi_category}")
                    
                    with col2:
                        st.metric("BMR", f"{recommendation['bmr']:.0f} kcal/day")
                        st.write("Basal Metabolic Rate")
                    
                    with col3:
                        st.metric("Daily Calorie Needs", f"{recommendation['daily_calories']:.0f} kcal/day")
                        st.write("Based on activity level")
                    
                    # Display recommended diet plan
                    st.subheader(f"Recommended Diet Plan: {recommendation['plan']['name']}")
                    st.write(recommendation['plan']['description'])
                    
                    st.subheader("Daily Calorie Target")
                    st.write(f"{recommendation['plan']['daily_calories']:.0f} kcal/day")
                    
                    st.subheader("Meal Recommendations")
                    for meal in recommendation['plan']['recommendations']:
                        st.write(f"• {meal}")
                    
                    st.info("This is a general recommendation. Please consult with a healthcare professional or registered dietitian before making significant changes to your diet.")
                else:
                    st.error("Could not generate diet recommendation. Please try again.")

if __name__ == "__main__":
    main()