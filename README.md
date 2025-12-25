# TV Genre Classification 🎬

This project is a machine learning application that predicts the **primary genre** of a TV show or movie based on its plot description. Using natural language processing (NLP) techniques, the model learns patterns in text data to classify titles into genres such as *Crime TV Shows*, *Documentaries*, *Comedies*, and more.

The project demonstrates a complete **ML pipeline**, including data preprocessing, feature extraction, model training, evaluation, and inference.

---

## Project Motivation

Streaming platforms contain thousands of titles with textual descriptions. Automatically classifying these descriptions into genres can support:
- content organization
- recommendation systems
- analytics and discovery tools  

This project was built to explore **text classification** using real-world data and to practice building an end-to-end, reproducible machine learning workflow.

---

## Dataset

- **Source:** Netflix Movies and TV Shows dataset (Kaggle)
- **Size:** ~8,600 titles after cleaning
- **Features used:**  
  - `description` (text input)
  - `listed_in` (genre labels)

To simplify the problem, multi-genre labels were reduced to a **single primary genre** per title.

### Data Storage & ETL (SQL Layer)
- Raw Netflix data is first loaded into a SQLite database using a custom ETL script.
- SQL queries are used to extract, filter, and transform structured records before model training.
- This approach separates data storage from model logic and mirrors real-world analytics pipelines.

## Database Schema

The project uses a lightweight SQLite database to store cleaned Netflix titles.

**Table: `titles`**
- `id` (INTEGER, primary key)
- `title` (TEXT)
- `description` (TEXT)
- `genre` (TEXT)
- `release_year` (INTEGER)
- `type` (TEXT)

The database is generated locally and excluded from version control for reproducibility.


---

## How It Works

### 1. Data Preprocessing
- Removed rows with missing descriptions or genres
- Extracted the primary genre from multi-label entries
- Filtered rare genres to improve model stability

### 2. Feature Engineering
- Converted text descriptions into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Removed common stop words to focus on informative terms

### 3. Model Training
- Trained a **multiclass Logistic Regression** classifier using **scikit-learn**
- Split data into training and test sets to evaluate generalization

### 4. Evaluation
- Achieved ~42% accuracy across 21 genres
- Analyzed performance using precision, recall, F1-score, and a confusion matrix

### 5. Inference
- Saved the trained pipeline using `joblib`
- Implemented a CLI script to predict genres for new descriptions

---

## Example Prediction

```bash
python src/predict.py "A detective investigates a series of mysterious murders in a small town."

Predicted genre: Crime TV Shows

Top 3 predictions:
- Crime TV Shows (0.21)
- British TV Shows (0.10)
- International Movies (0.08)
```
## Technologies Used
- Python
- scikit-learn
- Pandas
- NumPy
- joblib

## Run Locally
1. **Install dependencies**
`pip install -r requirements.txt`

2. **Load dataset into SQLite database**
` python src/load_to_db.py`
3. **Train model from SQL Database**
`python src/train.py`

4. **Run Predictions**
`python src/predict.py "Your description here"`

## Contact

If you’d like to connect or have questions about this project:

- **GitHub:** https://github.com/shanzayc
- **LinkedIn:** https://www.linkedin.com/in/shanzaychaudhry/
- **Email:** shanzayc@outlook.com

Feel free to reach out!



