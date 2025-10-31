Plantify: AI-Powered Crop Recommendation & Analysis Hub 🌾Plantify is an intelligent web application designed to help farmers, agricultural businesses, and scientific researchers make data-driven decisions. It leverages a high-accuracy machine learning model to recommend the most suitable crops based on soil and climate conditions. The app also features a suite of tools for in-depth data analysis and exploration.(This README is based on the project files: final_streamlit.py, Plantify_Notebook.ipynb, Crop_Dataset_updated.csv, and Plantify Project Documentation.pdf)✨ Key FeaturesSmart Crop Recommendation: Get instant, personalized crop recommendations by inputting soil parameters (Nitrogen, Phosphorus, Potassium, pH) and climate conditions (Temperature, Humidity, Rainfall).High-Accuracy Model: Powered by a Random Forest Classifier with 99.5%+ accuracy, ensuring reliable and trustworthy predictions.Role-Based Access: A tailored user experience for different user types:Business User: Access to the core Crop Recommendation and AI Chat features.Scientific Researcher: Full access to all tools, including Data Overview and advanced Plotting capabilities.Interactive AI Chat: Upload your own agricultural dataset (.csv) and ask questions in natural language. Powered by Google's Gemini API, the AI assistant can provide insights, generate code snippets, and summarize your data.Comprehensive Data Analysis:Data Overview: Quickly view your dataset's head, statistical summary (describe), and data types (info).Dynamic Plotting: Generate interactive visualizations on-the-fly, including histograms, box plots, scatter plots, and bar charts using Plotly.🛠️ Technology StackBackend & ML: Python, Pandas, NumPy, Scikit-learnFrontend: StreamlitData Visualization: Matplotlib, Plotly ExpressAI Chat: Google Gemini APIModel Development: Jupyter Notebook⚙️ Setup and InstallationTo run this project locally, follow these steps:1. Clone the Repositorygit clone [https://github.com/your-username/plantify.git](https://github.com/your-username/plantify.git)
cd plantify
2. Create and Activate a Virtual EnvironmentIt's recommended to use a virtual environment to manage dependencies.# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesCreate a file named requirements.txt and add the following lines to it:streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
google-generativeai
Then, install the packages using pip:pip install -r requirements.txt
4. Configure API KeyThe AI Chat feature requires a Google Gemini API key.Create a folder named .streamlit in your project's root directory.Inside this folder, create a file named secrets.toml.Add your API key to this file as follows:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
5. Run the ApplicationLaunch the Streamlit app from your terminal:streamlit run final_streamlit.py
The application should now be running in your web browser!📂 Project Structure.
├── .streamlit/
│   └── secrets.toml        # API keys and secrets
├── final_streamlit.py      # Main Streamlit application script
├── model.pkl               # Pre-trained Random Forest model (generated from notebook)
├── Crop_Dataset_updated.csv # Dataset used for training and analysis
├── Plantify_Notebook.ipynb # Jupyter Notebook for model development
├── requirements.txt        # Python dependencies
└── README.md               # This file
🧠 Model DetailsThe crop recommendation engine uses a Random Forest Classifier model trained on the Crop_Dataset_updated.csv.Features: N, P, K, temperature, humidity, ph, rainfallTarget: label (22 unique crop types)Performance: The model achieved an accuracy of 99.55% on the test set, demonstrating excellent performance in predicting the correct crop type. The full data cleaning, analysis, and model training process can be reviewed in the Plantify_Notebook.ipynb file.👥 Project ContributorsThis project was developed by:Marwan Mostafa IsmailFares Emad Abdelnaby ElsayedYouhanna George Jacoub IbrahimMohamed Hisham Hussian MahmoudAdham Jehad Abdelmoneam ElorabyMohamed Ezzat Saad Ghoraba📄 LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
