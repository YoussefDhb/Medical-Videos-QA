"""
eda_medvidqa.py - Exploratory Data Analysis for MedVidQA Dataset

This module provides comprehensive exploratory data analysis (EDA) for the MedVidQA cleaned
train, test, and validation datasets. It generates a variety of visualizations and statistical
summaries to understand the dataset characteristics and identify patterns.

The analysis is divided into three main categories:
1. Univariate Analysis: Examines individual variables in isolation
2. Bivariate Analysis: Explores relationships between pairs of variables
3. Multivariate Analysis: Analyzes interactions between multiple variables

Key Features:
-------------
- Video Analysis: Distribution of questions across videos, identification of top videos
- Temporal Analysis: Distribution of answer timestamps within videos
- Question Analysis: Length distribution, duplicate detection
- Correlation Analysis: Relationship between mutliple feature variables

Visualizations Generated:
------------------------
- Bar plots: Top 20 videos by question count
- Histograms: Video lengths, question lengths, questions per video
- Heatmaps: Correlation matrix
- Scatter plots: Video length vs number of questions correlation
- Temporal distributions: Answer start and end times

Output:
-------
All visualizations are saved as PNG files in the 'EDA/' directory with descriptive filenames
indicating the analysis type and dataset split (train/test/validation).

Directory Structure:
-------------------
- Input: MedVidQA_cleaned/ (cleaned JSON files for train, test, val)
- Output: EDA/ (generated visualization PNG files)

Dependencies:
------------
- pandas: Data manipulation and analysis
- matplotlib: Base plotting library
- seaborn: Statistical visualization
- json: Loading dataset files

Usage:
------
Run this script directly to generate all EDA visualizations:
    $ python eda_medvidqa.py

The script will process all three dataset splits (train, test, validation) and generate
comprehensive visualizations and statistics for each.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CLEANED_DIR = "MedVidQA_cleaned"
DATASET_DIR = "MedVidQA"
TRAIN_PATH = os.path.join(CLEANED_DIR, "train_cleaned.json")
TEST_PATH = os.path.join(CLEANED_DIR, "test_cleaned.json")
VAL_PATH = os.path.join(CLEANED_DIR, "val_cleaned.json")

ORIGINAL_PATHS = {
    'Train': os.path.join(DATASET_DIR, "train.json"),
    'Test': os.path.join(DATASET_DIR, "test.json"),
    'Validation': os.path.join(DATASET_DIR, "val.json")
}

EDA_DIR = "EDA"
os.makedirs(EDA_DIR, exist_ok=True)

def load_df(path):
    """
    Load a cleaned MedVidQA dataset from JSON file into a pandas DataFrame.

    Args:
        path (str): Path to the JSON file containing the cleaned dataset.

    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset with columns such as
                      video_id, question, answer, video_length, answer_start_second, etc.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_df = load_df(TRAIN_PATH)
test_df = load_df(TEST_PATH)
val_df = load_df(VAL_PATH)

# --- Visualization Functions ---
def plot_video_availability(set_name, cleaned_df):
    """
    Analyze and visualize video availability by comparing the original dataset with the cleaned one.
    
    Identifies how many videos and questions were lost due to being private, deleted, or 
    otherwise unavailable on YouTube.
    
    Args:
        set_name (str): Name of the dataset split ('Train', 'Test', 'Validation').
        cleaned_df (pd.DataFrame): The cleaned DataFrame currently being analyzed.
    """
    orig_path = ORIGINAL_PATHS.get(set_name)
    if not orig_path or not os.path.exists(orig_path):
        print(f"Skipping availability analysis for {set_name}: Original file not found.")
        return

    with open(orig_path, "r") as f:
        orig_data = json.load(f)
    orig_df = pd.DataFrame(orig_data)

    # Calculate metrics for Questions
    total_q = len(orig_df)
    available_q = len(cleaned_df)
    unavailable_q = total_q - available_q

    # Calculate metrics for Videos
    total_v = orig_df['video_id'].nunique()
    available_v = cleaned_df['video_id'].nunique()
    unavailable_v = total_v - available_v

    # Visualization 1: Question Availability (Pie Chart)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.pie([available_q, unavailable_q], labels=['Available', 'Unavailable'], 
            autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=140, explode=(0.1, 0))
    plt.title(f'Question Availability ({set_name})')

    # Visualization 2: Video Availability (Pie Chart)
    plt.subplot(1, 2, 2)
    plt.pie([available_v, unavailable_v], labels=['Available', 'Unavailable'], 
            autopct='%1.1f%%', colors=['#2196F3', '#FF9800'], startangle=140, explode=(0.1, 0))
    plt.title(f'Unique Video Availability ({set_name})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_availability_{set_name.lower()}.png'))
    plt.close()

    print(f"\n--- Availability Analysis: {set_name} ---")
    print(f"Questions: {available_q}/{total_q} available ({available_q/total_q:.2%})")
    print(f"Videos:    {available_v}/{total_v} available ({available_v/total_v:.2%})")
    print(f"Lost:      {unavailable_v} videos ({unavailable_q} questions) due to unavailability.")

def plot_top_videos_by_questions(df, set_name):
    """
    Generate a bar plot showing the top 20 videos by number of questions.

    Creates a horizontal bar plot displaying which videos have the most questions associated
    with them, helping identify videos with rich Q&A content. Also prints summary statistics
    including total questions, unique videos, and the top 10 videos by question count.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' column.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_questions_per_video_{set_name}.png'
        - Prints summary statistics to console

    Returns:
        None
    """
    plt.figure(figsize=(12,6))
    q_per_video = df['video_id'].value_counts()
    sns.barplot(x=q_per_video.index[:20], y=q_per_video.values[:20])
    plt.xticks(rotation=90)
    plt.title(f'Top 20 Videos by Number of Questions ({set_name} Set)')
    plt.xlabel('Video ID')
    plt.ylabel('Number of Questions')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_questions_per_video_{set_name.lower()}.png'))
    plt.close()
    print(f'\n{set_name} set:')
    print('Total questions:', len(df))
    print('Unique videos:', df["video_id"].nunique())
    print('Questions per video (top 10):')
    print(q_per_video.head(10))


def plot_video_length_distribution(df, set_name):
    """
    Generate a histogram showing the distribution of video lengths.

    Creates a histogram of video durations (in seconds) using only unique videos to avoid
    counting the same video multiple times. Provides insights into the typical length of
    videos in the dataset and prints descriptive statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' and 'video_length' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_video_length_distribution_{set_name}.png'
        - Prints descriptive statistics (mean, std, min, max, quartiles)

    Returns:
        None

    Note:
        Only processes data if 'video_length' column is present in the DataFrame.
    """
    if 'video_length' in df.columns:
        unique_videos = df.drop_duplicates(subset=['video_id'])
        plt.figure(figsize=(8,5))
        sns.histplot(unique_videos['video_length'], bins=30)
        plt.title(f'Distribution of Video Lengths (seconds) - {set_name} Set (Unique Videos)')
        plt.xlabel('Video Length (s)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_video_length_distribution_{set_name.lower()}.png'))
        plt.close()
        print('Video length stats (unique videos):')
        print(unique_videos['video_length'].describe())


def print_summary_stats(df, set_name):
    """
    Print comprehensive summary statistics for the dataset.

    Displays key metrics including total question count, unique video count, top videos
    by question count, and video length statistics if available.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Prints to console:
        - Total number of questions
        - Number of unique videos
        - Top 10 videos by question count
        - Video length statistics (if 'video_length' column exists)

    Returns:
        None
    """
    print(f'\n{set_name} set summary:')
    print('Total questions:', len(df))
    print('Unique videos:', df["video_id"].nunique())
    q_per_video = df['video_id'].value_counts()
    print('Questions per video (top 10):')
    print(q_per_video.head(10))
    if 'video_length' in df.columns:
        print('Video length stats:')
        print(df['video_length'].describe())


def plot_question_length_distribution(df, set_name):
    """
    Generate a histogram showing the distribution of question lengths in words.

    Analyzes and visualizes how long questions typically are by counting the number of words
    in each question. This helps understand question complexity and verbosity patterns.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'question' column.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_question_length_distribution_{set_name}.png'
        - Prints descriptive statistics for question lengths

    Returns:
        None

    Note:
        - Only processes data if 'question' column is present in the DataFrame.
        - Question length is measured in number of words (whitespace-separated tokens).
    """
    if 'question' in df.columns:
        lengths = df['question'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(8,5))
        sns.histplot(lengths, bins=30)
        plt.title(f'Distribution of Question Lengths ({set_name} Set)')
        plt.xlabel('Question Length (words)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_question_length_distribution_{set_name.lower()}.png'))
        plt.close()
        print(f'Question length stats ({set_name}):')
        print(lengths.describe())


def plot_questions_per_video_hist(df, set_name):
    """
    Generate a histogram showing the distribution of questions per video.

    Creates a histogram that shows how many questions each video typically has, revealing
    whether questions are distributed evenly or concentrated on certain videos.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' column.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_questions_per_video_hist_{set_name}.png'
        - Prints descriptive statistics for questions per video distribution

    Returns:
        None

    Note:
        This visualization helps identify if some videos are over-represented in the dataset.
    """
    q_per_video = df['video_id'].value_counts()
    plt.figure(figsize=(8,5))
    sns.histplot(q_per_video, bins=30)
    plt.title(f'Questions per Video Distribution ({set_name} Set)')
    plt.xlabel('Questions per Video')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_questions_per_video_hist_{set_name.lower()}.png'))
    plt.close()
    print(f'Questions per video stats ({set_name}):')
    print(q_per_video.describe())

def plot_temporal_distribution(df, set_name):
    """
    Generate overlapping histograms showing temporal distribution of answer timestamps.

    Visualizes when answers occur within videos by plotting the distribution of answer start
    and end times. This helps understand if answers are concentrated at certain time points
    in the videos (e.g., beginning, middle, or end).

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'answer_start_second' and
                          'answer_end_second' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_temporal_distribution_{set_name}.png'
        - Prints descriptive statistics for answer start and end times

    Returns:
        None

    Note:
        - Only processes data if 'answer_start_second' and 'answer_end_second' columns exist.
        - If string-based 'answer_start' and 'answer_end' columns exist instead, prints sample values.
    """
    if 'answer_start_second' in df.columns and 'answer_end_second' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df['answer_start_second'], bins=30, color='blue', label='Start')
        sns.histplot(df['answer_end_second'], bins=30, color='orange', label='End', alpha=0.6)
        plt.title(f'Temporal Distribution of Answers ({set_name} Set)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Count')
        plt.legend(['Answer Start', 'Answer End'])
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_temporal_distribution_{set_name.lower()}.png'))
        plt.close()
        print(f'Answer start/end stats ({set_name}):')
        print('Start:', df['answer_start_second'].describe())
        print('End:', df['answer_end_second'].describe())
    elif 'answer_start' in df.columns and 'answer_end' in df.columns:
        print(f'Answer start/end columns present as strings in {set_name} set.')
        print('Sample start:', df['answer_start'].head())
        print('Sample end:', df['answer_end'].head())


def plot_correlation_video_length_questions(df, set_name):
    """
    Generate a scatter plot showing the correlation between video length and question count.

    Creates a scatter plot to visualize the relationship between how long a video is and how
    many questions are associated with it. Computes and displays the correlation coefficient
    to quantify the strength of this relationship.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'video_id' and 'video_length' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_video_length_vs_questions_{set_name}.png'
        - Prints correlation matrix between video length and number of questions

    Returns:
        None

    Note:
        - Only processes data if 'video_length' column is present.
        - Positive correlation suggests longer videos tend to have more questions.
    """
    if 'video_length' in df.columns:
        q_per_video = df.groupby('video_id').size().reset_index(name='num_questions')
        video_lengths = df.drop_duplicates('video_id')[['video_id', 'video_length']]
        merged = pd.merge(q_per_video, video_lengths, on='video_id')
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='video_length', y='num_questions', data=merged)
        plt.title(f'Video Length vs. Number of Questions ({set_name} Set)')
        plt.xlabel('Video Length (s)')
        plt.ylabel('Number of Questions')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_video_length_vs_questions_{set_name.lower()}.png'))
        plt.close()
        print(f'Correlation (video length vs. questions) ({set_name}):')
        print(merged[['video_length', 'num_questions']].corr())


def plot_duplicate_questions(df, set_name):
    """
    Generate a bar plot showing the proportion of unique vs. duplicate questions.

    Identifies and visualizes duplicate questions within the same video, helping assess
    data quality and identify potential issues with question redundancy.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with 'question' and 'video_id' columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_duplicate_questions_{set_name}.png'
        - Prints count and percentage of duplicate questions

    Returns:
        None

    Note:
        - Only processes data if 'question' column is present.
        - Duplicates are identified by matching both question text and video_id.
        - High duplication rates may indicate data quality issues.
    """
    if 'question' in df.columns:
        dupes = df.duplicated(subset=['question', 'video_id'], keep=False)
        num_dupes = dupes.sum()
        total = len(df)
        plt.figure(figsize=(6,4))
        sns.barplot(x=['Unique', 'Duplicate'], y=[total-num_dupes, num_dupes])
        plt.title(f'Unique vs. Duplicate Questions ({set_name} Set)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f'eda_duplicate_questions_{set_name.lower()}.png'))
        plt.close()
        print(f'Duplicate questions ({set_name}): {num_dupes} / {total} ({num_dupes/total:.2%})')


def plot_correlation_matrix(df, set_name):
    """
    Generate a correlation matrix heatmap showing relationships between multiple numerical variables.

    Creates a comprehensive correlation matrix analyzing relationships between video length,
    number of questions per video, average question length, answer start/end times, and
    answer duration. This true multivariate analysis reveals complex interdependencies
    between multiple features simultaneously.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with numerical columns.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        - Saves a PNG file: 'EDA/eda_correlation_matrix_{set_name}.png'
        - Prints the correlation matrix values

    Returns:
        None

    Note:
        - Requires 'video_length' column to be present.
        - Computes derived features: num_questions, avg_question_length, answer_duration.
        - Uses coolwarm colormap with annotations for clear interpretation.
        - Correlation values range from -1 (negative) to +1 (positive correlation).
    """
    if 'video_length' not in df.columns:
        print(f'Skipping correlation matrix for {set_name}: video_length column not present')
        return

    # Prepare multivariate features
    video_features = []
    for video_id in df['video_id'].unique():
        video_data = df[df['video_id'] == video_id]

        features = {
            'video_length': video_data['video_length'].iloc[0],
            'num_questions': len(video_data)
        }

        # Add question length if available
        if 'question' in df.columns:
            features['avg_question_length'] = video_data['question'].apply(
                lambda x: len(str(x).split())
            ).mean()

        # Add temporal features if available
        if 'answer_start_second' in df.columns:
            features['avg_answer_start'] = video_data['answer_start_second'].mean()

        if 'answer_end_second' in df.columns:
            features['avg_answer_end'] = video_data['answer_end_second'].mean()

        # Calculate answer duration if both start and end are available
        if 'answer_start_second' in df.columns and 'answer_end_second' in df.columns:
            features['avg_answer_duration'] = (
                video_data['answer_end_second'] - video_data['answer_start_second']
            ).mean()

        video_features.append(features)

    features_df = pd.DataFrame(video_features)

    # Compute correlation matrix
    correlation_matrix = features_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title(f'Correlation Matrix - Multivariate Analysis ({set_name} Set)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f'eda_correlation_matrix_{set_name.lower()}.png'), dpi=300)
    plt.close()

    print(f'\nCorrelation Matrix ({set_name} Set):')
    print(correlation_matrix)
    print(f'\nNumber of variables analyzed: {len(correlation_matrix.columns)}')
    print(f'Features: {", ".join(correlation_matrix.columns)}')

# --- Univariate Analysis ---
def univariate_analysis(df, set_name):
    """
    Perform comprehensive univariate analysis on the dataset.

    Executes all single-variable analyses to understand the distribution and characteristics
    of individual features in isolation. This includes analyzing video distributions, question
    characteristics, and basic dataset statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset to analyze.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Generates multiple visualizations and prints summary statistics:
        - Video availability (compared to original)
        - Top videos by question count
        - Video length distribution
        - Question length distribution
        - Questions per video histogram
        - Summary statistics

    Returns:
        None
    """
    plot_video_availability(set_name, df)
    plot_top_videos_by_questions(df, set_name)
    plot_video_length_distribution(df, set_name)
    plot_question_length_distribution(df, set_name)
    plot_questions_per_video_hist(df, set_name)
    print_summary_stats(df, set_name)

# --- Bivariate Analysis ---
def bivariate_analysis(df, set_name):
    """
    Perform comprehensive bivariate analysis on the dataset.

    Examines relationships between pairs of variables to understand how different features
    interact with each other. This includes temporal patterns, correlations, and data quality
    issues like duplicates.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset to analyze.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Generates multiple visualizations and prints statistics:
        - Temporal distribution of answers within videos
        - Correlation between video length and question count

    Returns:
        None
    """
    plot_temporal_distribution(df, set_name)
    plot_correlation_video_length_questions(df, set_name)

# --- Multivariate Analysis ---
def multivariate_analysis(df, set_name):
    """
    Perform comprehensive multivariate analysis on the dataset.

    Analyzes complex interactions between multiple variables (3+) simultaneously to reveal
    patterns that may not be apparent in univariate or bivariate analyses. This includes
    correlation analysis between video characteristics, question patterns, and temporal features.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset to analyze.
        set_name (str): Name of the dataset split (e.g., 'Train', 'Test', 'Validation').

    Output:
        Generates multivariate visualizations:
        - Correlation matrix heatmap: Relationships between video_length, num_questions,
          avg_question_length, avg_answer_start, avg_answer_end, and avg_answer_duration

    Returns:
        None

    Note:
        True multivariate analysis examines 3 or more variables simultaneously. The correlation
        matrix provides a comprehensive view of all pairwise relationships among multiple features.
    """
    plot_correlation_matrix(df, set_name)

def main():
    """
    Main entry point for the EDA script.

    Orchestrates the complete exploratory data analysis workflow by loading all three dataset
    splits (train, test, validation) and performing univariate, bivariate, and multivariate
    analyses on each. Results are saved to the EDA directory.

    Workflow:
        1. Load cleaned datasets (train, test, validation)
        2. For each dataset split:
           a. Perform univariate analysis
           b. Perform bivariate analysis
           c. Perform multivariate analysis
        3. Save all visualizations to EDA/ directory

    Args:
        None

    Returns:
        None

    Note:
        This function assumes that cleaned JSON files exist in the MedVidQA_cleaned/ directory.
    """
    # --- Run EDA for all sets ---
    for df, name in zip([train_df, test_df, val_df], ['Train', 'Test', 'Validation']):
        print(f"\n{'='*20} {name} Set: Univariate Analysis {'='*20}")
        univariate_analysis(df, name)
        print(f"\n{'='*20} {name} Set: Bivariate Analysis {'='*20}")
        bivariate_analysis(df, name)
        print(f"\n{'='*20} {name} Set: Multivariate Analysis {'='*20}")
        multivariate_analysis(df, name)

if __name__ == "__main__":
    main()