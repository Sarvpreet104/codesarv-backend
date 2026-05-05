import pandas as pd

def take_one_sample(test_data_path):
    
    df = pd.read_csv(test_data_path)
    sample = df.sample(n=1)

    return sample.to_dict(orient="records")[0]

if __name__ == '__main__':
    sample = take_one_sample("binary-logistic-regression")
    print(sample)

    