import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging

import torch._dynamo
torch._dynamo.config.suppress_errors = True

logging.basicConfig(
    filename="mbert_embedding.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    logging.info("Model and tokenizer loaded successfully.")

    df = pd.read_csv("fraud_text.csv")
    df = df.dropna(subset=['mda'])
    logging.info(f"Dataset loaded. Shape: {df.shape}")

    def get_embeddings(text_list, batch_size=16):
        logging.info(f"Generating embeddings for {len(text_list)} texts.")
        all_embeddings = []

        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{(len(text_list) - 1) // batch_size + 1}")

            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_embeddings.append(cls_embeddings)

            torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    # Generate and attach embeddings
    mda_embeddings = get_embeddings(df['mda'].tolist())
    embedding_df = pd.DataFrame(
        mda_embeddings,
        index=df.index,
        columns=[f'mbert_full_cls_{i}' for i in range(mda_embeddings.shape[1])]
    )

    df_combined = pd.concat([df, embedding_df], axis=1)
    df_combined.to_csv("fraud_mbert_full.csv", index=False)
    logging.info("Saved full dataset with BERT embeddings to df_mbert_full.csv.")

except Exception as e:
    logging.error(f"Error encountered: {e}", exc_info=True)
