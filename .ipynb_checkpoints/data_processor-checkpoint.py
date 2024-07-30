import pandas as pd
import re
from dataclasses import dataclass
from typing import Optional
from configs import Config


@dataclass
class QuestionData:
    question: str
    rag_rain_comp: Optional[str]
    rag_rain_dishon: Optional[str]
    rag: Optional[str]


@dataclass
class MatchingData:
    question_data: QuestionData
    match_idx: int
    matched_row: pd.Series


def extract_text_after_keyword(text: str, keyword: str = 'Assistant:') -> str:
    match = re.search(rf'{keyword}(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else ''


def load_and_prepare_data() -> pd.DataFrame:
    # Load the TSV file
    df = pd.read_csv(f'{Config.DATA_DIR}/json_files_to_tsv_loaded.tsv', sep="\t")
    df['Rag'] = df['pa'].apply(extract_text_after_keyword)

    # Drop duplicates and reset index
    df = df.drop_duplicates(subset=['Question']).reset_index(drop=True)
    df = df[['Question', 'Rag_Rain_comp', 'Rag_Rain_dishon', 'Rag']]
    return df


def load_privacyQA_data() -> pd.DataFrame:
    # Load the PrivacyQA CSV file
    return pd.read_csv(f'{Config.DATA_DIR}/PrivacyQA.csv', sep=";")


def find_matching_row(question: str, df2: pd.DataFrame) -> Optional[int]:
    for idx, row in df2.iterrows():
        if re.search(re.escape(question), row['Questions'], re.IGNORECASE):
            return idx
    return None


def merge_data(df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Find matching indices
    df['match_idx'] = df['Question'].apply(lambda q: find_matching_row(q, df2))

    # Remove rows with no matches
    df = df.dropna(subset=['match_idx'])

    # Convert match_idx to integer
    df['match_idx'] = df['match_idx'].astype(int)

    # Merge dataframes
    merged_df = pd.merge(df, df2, left_on='match_idx', right_index=True, how='left')

    # Drop the extra columns used for merging
    merged_df = merged_df.drop(columns=['match_idx'])

    # Drop specified columns if they exist
    columns_to_drop = ['Unnamed: 5', 'Rag Question']
    merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns], errors='ignore')

    return merged_df


def main():
    df = load_and_prepare_data()
    df2 = load_privacyQA_data()
    merged_df = merge_data(df, df2)

    # Save the cleaned data to a TSV file
    merged_df.to_csv(f'{Config.DATA_DIR}/processed_data.tsv', sep='\t', index=False)
    print(f"Processed data saved to {Config.DATA_DIR}/processed_data.tsv")


if __name__ == "__main__":
    main()
