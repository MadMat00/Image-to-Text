import pandas as pd
import requests

def load_data():
    data = pd.read_csv('data/image_prompts_df.csv', index_col=0)
    data.index.name = 'ID'
    data.reset_index(inplace=True)
    data.drop(columns=['Date'], inplace=True)
    return data

def download_images(data):
    print(data.head())
    for index, row in data.iterrows():
        try:
            response = requests.get(row['Attachments'], stream=True)
            if response.status_code == 200:
                ext = row['Attachments'].split('.')[-1]
                with open(f'data/images/{row["ID"]}.{ext}', 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
        except KeyboardInterrupt:
            break
        except:
            print(f'Image {row["ID"]} not found')

def main():
    data = load_data()
    download_images(data)

if __name__ == '__main__':
    main()