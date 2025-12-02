import argparse
import pandas as pd
from src.model import load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", default="predictions.csv")
    args = parser.parse_args()

    model = load_model(args.model_path)
    df = pd.read_csv(args.input-file)

    predictions = model.predict(df)

    out = pd.DataFrame({"prediction": predictions})
    out.to_csv(args.output-file, index=False)

    print("Saved predictions to:", args.output-file)

if __name__ == "__main__":
    main()
