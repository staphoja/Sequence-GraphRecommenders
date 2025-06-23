# Team JACKS – Final Project Usage

## 1. Install dependencies

First, install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 2. Project structure

- `checkpoint1.py`: Recommenders from Checkpoint 1  
- `checkpoint2.py`: Recommenders from Checkpoint 2  
- `checkpoint3.py`: Recommenders from Checkpoint 3  

---

## 3. Run visual comparison of models

To compare all models from the checkpoints, run the following command from the **main directory**:

```bash
uv run recommender_analysis_visualization.py
```

---

## 4. Final model and analysis

The folder `final/` contains:

- `final_recommender.py`: Final recommendation model  
- `final_analysis_visualization.py`: Visualization and evaluation of the final model  

To run the final model's visual analysis, use:

```bash
uv run final/final_analysis_visualization.py
```

---

## 5. Leaderboard submission

The file `submission.py` contains the finalized model ready for leaderboard submission.
