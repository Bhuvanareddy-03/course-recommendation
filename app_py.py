import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 📥 Upload and preprocess dataset
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df = df[['user_id', 'course_id', 'course_name', 'instructor', 'difficulty_level',
             'course_duration_hours', 'certification_offered', 'study_material_available',
             'course_price', 'feedback_score', 'rating']]
    
    df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
    df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})
    df['difficulty_level'] = df['difficulty_level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})
    
    scaler = MinMaxScaler()
    df[['course_duration_hours', 'course_price', 'feedback_score']] = scaler.fit_transform(
        df[['course_duration_hours', 'course_price', 'feedback_score']]
    )
    return df

# 📊 Accuracy evaluation for collaborative model
def evaluate_collaborative_accuracy(df, course_avg):
    test_df = df.sample(frac=0.2, random_state=42)
    test_df['predicted_rating'] = test_df['course_id'].map(course_avg)
    test_df = test_df.dropna(subset=['predicted_rating'])
    rmse = np.sqrt(mean_squared_error(test_df['rating'], test_df['predicted_rating']))
    mae = mean_absolute_error(test_df['rating'], test_df['predicted_rating'])
    return rmse, mae

# 📊 Accuracy evaluation for content-based model
def evaluate_content_accuracy(df, course_sim_df, course_avg):
    test_df = df.sample(frac=0.2, random_state=42)
    predictions = []
    for _, row in test_df.iterrows():
        cid = row['course_id']
        if cid in course_sim_df.index:
            similar = course_sim_df[cid].sort_values(ascending=False)[1:6].index
            avg_sim_rating = df[df['course_id'].isin(similar)]['rating'].mean()
            predictions.append(avg_sim_rating)
        else:
            predictions.append(course_avg.get(cid, 3.0))
    rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
    mae = mean_absolute_error(test_df['rating'], predictions)
    return rmse, mae

# 🔍 Content-based recommendation
def recommend_similar_courses(df, course_sim_df, course_id, top_n=5):
    if course_id not in course_sim_df.index:
        return pd.DataFrame()
    similar = course_sim_df[course_id].sort_values(ascending=False)[1:top_n+1]
    return df[df['course_id'].isin(similar.index)][['course_name', 'instructor', 'difficulty_level']]

# 🔍 Collaborative recommendation
def recommend_top_courses(df, course_avg, user_id, top_n=5):
    rated = df[df['user_id'] == user_id]['course_id'].tolist()
    unrated = course_avg[~course_avg.index.isin(rated)].sort_values(ascending=False).head(top_n)
    return df[df['course_id'].isin(unrated.index)][['course_name', 'instructor', 'difficulty_level']]

# 📈 Visual comparison
def plot_accuracy_comparison(rmse_cb, mae_cb, rmse_cf, mae_cf):
    models = ['Content-Based', 'Collaborative']
    rmse_values = [rmse_cb, rmse_cf]
    mae_values = [mae_cb, mae_cf]
    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width/2 for i in x], rmse_values, width, label='RMSE', color='skyblue')
    ax.bar([i + width/2 for i in x], mae_values, width, label='MAE', color='salmon')
    ax.set_xlabel('Recommendation Model')
    ax.set_ylabel('Error Value')
    ax.set_title('RMSE vs MAE for Recommendation Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    st.pyplot(fig)

# 🎯 Streamlit UI
st.title("🎓 Online Course Recommendation System")

uploaded_file = st.file_uploader("Upload your Excel dataset (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)

    course_features = df.drop_duplicates('course_id')[[
        'course_id', 'difficulty_level', 'course_duration_hours',
        'certification_offered', 'study_material_available',
        'course_price', 'feedback_score'
    ]].set_index('course_id')

    course_sim = cosine_similarity(course_features)
    course_sim_df = pd.DataFrame(course_sim, index=course_features.index, columns=course_features.index)
    course_avg = df.groupby('course_id')['rating'].mean()

    st.markdown("Choose Recommendation Method")
    model_choice = st.radio("Select a model:", ["Content-Based", "Collaborative"])

    st.markdown("Evaluate Accuracy")
    if st.button("Evaluate Accuracy"):
        rmse_cb, mae_cb = evaluate_content_accuracy(df, course_sim_df, course_avg)
        rmse_cf, mae_cf = evaluate_collaborative_accuracy(df, course_avg)

        st.success("✅ Accuracy Results")
        st.write(f"**Content-Based Filtering**\n- RMSE: {rmse_cb:.3f}\n- MAE: {mae_cb:.3f}")
        st.write(f"**Collaborative Filtering**\n- RMSE: {rmse_cf:.3f}\n- MAE: {mae_cf:.3f}")

        st.markdown("### 📊 Accuracy Comparison Chart")
        plot_accuracy_comparison(rmse_cb, mae_cb, rmse_cf, mae_cf)

        st.markdown("### 🏆 Best Performing Method")
        best_model = "Content-Based" if rmse_cb < rmse_cf else "Collaborative"
        st.info(f"Based on RMSE, the best method is: **{best_model} Filtering**")

    st.markdown("Get Recommendations")
    if model_choice == "Content-Based":
        course_id = st.number_input("Enter a Course ID you liked:", min_value=1)
        if st.button("Get Recommendations"):
            results = recommend_similar_courses(df, course_sim_df, course_id)
            st.write("Recommended Courses:")
            st.dataframe(results)
    else:
        user_id = st.number_input("Enter your User ID:", min_value=1)
        if st.button("Get Recommendations"):
            results = recommend_top_courses(df, course_avg, user_id)
            st.write("Recommended Courses:")
            st.dataframe(results)

else:
    st.warning("📂 Please upload a dataset to begin.")
